"""Multi-provider LLM factory.

Adds two production-grade behaviours on top of plain LangChain integrations:

1. **Proactive rate limiting** — a process-wide token bucket per provider
   throttles outbound calls so we don't spike past tier limits (e.g.
   Anthropic's ``30k input tokens/min`` on Sonnet 4.6).
2. **Inline retry with backoff** — transient errors (429s, 5xx, network)
   retry inside the same node instead of crashing the whole graph turn.

Slug format: ``provider/model`` (``openai/gpt-4o``,
``anthropic/claude-sonnet-4-6``).

``OPENAI_BASE_URL`` applies **only** to ``openai/...`` slugs. Roles still set to
``anthropic/...`` call Anthropic's API regardless of ``OPENAI_BASE_URL``.
"""
from __future__ import annotations

import logging
import threading

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from backend.config import get_settings

_logger = logging.getLogger(__name__)

# Module-wide singletons keyed by provider — shared across all agent calls.
_LIMITERS: dict[str, InMemoryRateLimiter] = {}
_LIMITER_LOCK = threading.Lock()


def _limiter_for(provider: str) -> InMemoryRateLimiter:
    settings = get_settings()
    with _LIMITER_LOCK:
        existing = _LIMITERS.get(provider)
        if existing is not None:
            return existing
        if provider == "anthropic":
            rps = settings.anthropic_requests_per_second
            burst = settings.anthropic_burst
        elif provider == "openai":
            rps = settings.openai_requests_per_second
            burst = settings.openai_burst
        else:
            rps, burst = 1.0, 2
        limiter = InMemoryRateLimiter(
            requests_per_second=rps,
            check_every_n_seconds=0.1,
            max_bucket_size=max(1, burst),
        )
        _LIMITERS[provider] = limiter
        return limiter


# Exception classes considered transient (provider 429s, server errors,
# overload signals, network blips). Imported lazily to avoid hard deps.
def _transient_exceptions() -> tuple[type[BaseException], ...]:
    excs: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:
        from anthropic import (  # type: ignore[import-not-found]
            APIConnectionError as AAPIConnectionError,
            APIStatusError as AAPIStatusError,
            APITimeoutError as AAPITimeoutError,
            RateLimitError as ARateLimitError,
        )

        excs.extend([AAPIConnectionError, AAPIStatusError, AAPITimeoutError, ARateLimitError])
    except Exception:
        pass
    try:
        from openai import (  # type: ignore[import-not-found]
            APIConnectionError as OAPIConnectionError,
            APIStatusError as OAPIStatusError,
            APITimeoutError as OAPITimeoutError,
            RateLimitError as ORateLimitError,
        )

        excs.extend([OAPIConnectionError, OAPIStatusError, OAPITimeoutError, ORateLimitError])
    except Exception:
        pass
    return tuple(excs)


def _wrap_with_retry(model: BaseChatModel) -> Runnable:
    """Attach inline retry-with-backoff to a chat model.

    The returned object is a ``Runnable`` (not a ``BaseChatModel``); call
    ``.bind_tools(...)`` on the *original* model first, then wrap the
    result with this helper.
    """
    settings = get_settings()
    return model.with_retry(
        retry_if_exception_type=_transient_exceptions(),
        stop_after_attempt=max(1, settings.llm_max_retries),
        wait_exponential_jitter=True,
    )


def _split_slug(model_slug: str) -> tuple[str, str]:
    if "/" not in model_slug:
        return "openai", model_slug
    p, n = model_slug.split("/", 1)
    return p.lower(), n


def log_resolved_llm_routing() -> None:
    """Log where each configured role sends traffic (local vs cloud).

    Misconfiguration: OPENAI_BASE_URL set for llama-server but LEAD_MODEL (etc.)
    still ``anthropic/...`` → those roles never hit localhost.
    """
    settings = get_settings()
    rows: list[tuple[str, str]] = [
        ("pm_model", settings.pm_model),
        ("researcher_model", settings.researcher_model),
        ("lead_model", settings.lead_model),
        ("dev_model", settings.dev_model),
        ("backend_dev_model", settings.backend_dev_model or settings.dev_model),
        ("frontend_dev_model", settings.frontend_dev_model or settings.dev_model),
        ("grader_model", settings.grader_model),
    ]
    base = (settings.openai_base_url or "").strip()
    for label, slug in rows:
        provider, model_name = _split_slug(slug)
        if provider == "openai" and base:
            target = f"OpenAI-compatible {base} (model={model_name!r})"
        elif provider == "openai":
            target = f"OpenAI platform API (model={model_name!r})"
        else:
            target = f"Anthropic API (model={model_name!r})"
        _logger.info("LLM routing %s=%s -> %s", label, slug, target)

    if base:
        conflicts = [label for label, slug in rows if _split_slug(slug)[0] == "anthropic"]
        if conflicts:
            _logger.warning(
                "OPENAI_BASE_URL is set but these env roles still use anthropic/ slugs %s — "
                "they will NOT reach your local server. Set each to openai/<model_id>.",
                conflicts,
            )


def get_chat_model(model_slug: str, *, temperature: float = 0.0, **kwargs) -> BaseChatModel:
    """Resolve a ``provider/model`` slug to a rate-limited chat model.

    Inline retry is *not* applied here so that callers can still call
    ``.bind_tools(...)`` (which only exists on ``BaseChatModel``). Use
    :func:`get_chat_model_with_retry` if you want the retry-wrapped
    variant after binding tools.
    """
    settings = get_settings()
    provider, name = _split_slug(model_slug)

    common: dict = {
        "temperature": temperature,
        "rate_limiter": _limiter_for(provider),
        "max_tokens": settings.max_output_tokens,
        **kwargs,
    }

    if provider == "openai":
        common.pop("max_tokens", None)  # ChatOpenAI uses ``max_tokens`` differently
        kwargs_openai: dict = {
            "model": name,
            "api_key": settings.openai_api_key or None,
            "max_retries": settings.llm_max_retries,
            "timeout": settings.openai_request_timeout,
            **common,
        }
        base = (settings.openai_base_url or "").strip()
        if base:
            kwargs_openai["base_url"] = base
            # Local OpenAI-compatible servers usually ignore the key but require a value.
            if not kwargs_openai.get("api_key"):
                kwargs_openai["api_key"] = "not-needed"
        return ChatOpenAI(**kwargs_openai)
    if provider == "anthropic":
        return ChatAnthropic(
            model=name,
            api_key=settings.anthropic_api_key or None,
            max_retries=settings.llm_max_retries,
            **common,
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Supported: openai, anthropic."
    )


def with_retry(runnable: Runnable) -> Runnable:
    """Public alias so callers can wrap a tool-bound model in retry logic."""
    return _wrap_with_retry(runnable)  # type: ignore[arg-type]


def pm_model() -> BaseChatModel:
    return get_chat_model(get_settings().pm_model)


def researcher_model() -> BaseChatModel:
    return get_chat_model(get_settings().researcher_model)


def lead_model() -> BaseChatModel:
    return get_chat_model(get_settings().lead_model)


def dev_model() -> BaseChatModel:
    return get_chat_model(get_settings().dev_model)


def backend_dev_model() -> BaseChatModel:
    s = get_settings()
    slug = s.backend_dev_model or s.dev_model
    return get_chat_model(slug)


def frontend_dev_model() -> BaseChatModel:
    s = get_settings()
    slug = s.frontend_dev_model or s.dev_model
    return get_chat_model(slug)


def grader_model() -> BaseChatModel:
    return get_chat_model(get_settings().grader_model)
