"""Multi-provider LLM factory.

Adds two production-grade behaviours on top of plain LangChain integrations:

1. **Proactive rate limiting** — a process-wide token bucket per provider
   throttles outbound calls so we don't spike past tier limits (e.g.
   Anthropic's ``30k input tokens/min`` on Sonnet 4.6).
2. **Inline retry with backoff** — transient errors (429s, 5xx, network)
   retry inside the same node instead of crashing the whole graph turn.

Slug format: ``provider/model`` (``openai/gpt-4o``,
``anthropic/claude-sonnet-4-6``).

``OPENAI_BASE_URL`` applies to ``openai/...`` slugs unless the researcher uses
``RESEARCHER_OPENAI_BASE_URL`` or ``RESEARCHER_USE_PLATFORM_OPENAI``. Roles set to
``anthropic/...`` call Anthropic's API regardless of ``OPENAI_BASE_URL``.
"""
from __future__ import annotations

import logging
import threading
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from backend.config import Settings, get_settings

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


OpenaiHostOverride = Literal["none", "researcher", "grader"]


def _openai_base_url_for_client(
    settings: Settings,
    *,
    host_override: OpenaiHostOverride = "none",
) -> str | None:
    """Return ``base_url`` for ``ChatOpenAI``, or ``None`` to omit (SDK default host).

    ``host_override`` applies role-specific OpenAI URL / platform flags before
    falling back to the global ``openai_base_url``.
    """
    if host_override == "researcher":
        explicit = (settings.researcher_openai_base_url or "").strip()
        if explicit:
            return explicit
        if settings.researcher_use_platform_openai:
            return None
    elif host_override == "grader":
        explicit = (settings.grader_openai_base_url or "").strip()
        if explicit:
            return explicit
        if settings.grader_use_platform_openai:
            return None
    global_base = (settings.openai_base_url or "").strip()
    return global_base or None


def log_resolved_llm_routing() -> None:
    """Log where each configured role sends traffic (local vs cloud).

    Misconfiguration: OPENAI_BASE_URL set for llama-server but LEAD_MODEL (etc.)
    still ``anthropic/...`` → those roles never hit localhost.
    """
    settings = get_settings()
    _openai_is_local: dict[str, bool] = {
        "pm_model": settings.pm_is_local,
        "researcher_model": settings.researcher_is_local,
        "lead_model": settings.lead_is_local,
        "dev_model": settings.dev_is_local,
        "backend_dev_model": settings.backend_dev_is_local,
        "frontend_dev_model": settings.frontend_dev_is_local,
        "grader_model": settings.grader_is_local,
    }
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
        if provider == "openai":
            eff_base: str | None
            if label == "researcher_model":
                eff_base = _openai_base_url_for_client(settings, host_override="researcher")
            elif label == "grader_model":
                eff_base = _openai_base_url_for_client(settings, host_override="grader")
            else:
                eff_base = _openai_base_url_for_client(settings, host_override="none")
            if eff_base:
                target = f"OpenAI-compatible {eff_base} (model={model_name!r})"
            else:
                target = f"OpenAI platform API (model={model_name!r})"
        else:
            target = f"Anthropic API (model={model_name!r})"
        if provider == "openai":
            target = f"{target}; IS_LOCAL={_openai_is_local[label]}"
        _logger.info("LLM routing %s=%s -> %s", label, slug, target)

    if base:
        conflicts = [label for label, slug in rows if _split_slug(slug)[0] == "anthropic"]
        if conflicts:
            _logger.warning(
                "OPENAI_BASE_URL is set but these env roles still use anthropic/ slugs %s — "
                "they will NOT reach your local server. Set each to openai/<model_id>.",
                conflicts,
            )


def get_chat_model(
    model_slug: str,
    *,
    temperature: float = 0.0,
    openai_host_override: OpenaiHostOverride = "none",
    **kwargs,
) -> BaseChatModel:
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
        kwargs_openai: dict = {
            "model": name,
            "api_key": settings.openai_api_key or None,
            "max_retries": settings.llm_max_retries,
            "timeout": settings.openai_request_timeout,
            "max_tokens": settings.max_output_tokens,
            **common,
        }
        base = _openai_base_url_for_client(
            settings,
            host_override=openai_host_override,
        )
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
    s = get_settings()
    provider, _ = _split_slug(s.researcher_model)
    return get_chat_model(
        s.researcher_model,
        openai_host_override="researcher" if provider == "openai" else "none",
    )


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
    s = get_settings()
    provider, _ = _split_slug(s.grader_model)
    return get_chat_model(
        s.grader_model,
        openai_host_override="grader" if provider == "openai" else "none",
    )
