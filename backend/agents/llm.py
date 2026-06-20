"""Multi-provider LLM factory.

Adds two production-grade behaviours on top of plain LangChain integrations:

1. **Proactive rate limiting** — a process-wide token bucket per provider
   throttles outbound calls so we don't spike past tier limits (e.g.
   Anthropic's ``30k input tokens/min`` on Sonnet 4.6).
2. **Inline retry with backoff** — transient errors (429s, 5xx, network)
   retry inside the same node instead of crashing the whole graph turn.

Slug format: ``provider/model`` (``openai/gpt-4o``,
``anthropic/claude-sonnet-4-6``, ``openrouter/anthropic/claude-sonnet-4``).

OpenRouter catalog ids often look like ``vendor/model`` (e.g. ``nex-agi/foo``).
Those are **not** gateway providers — with ``OPENROUTER_API_KEY`` set you can use
the OpenRouter id directly (``nex-agi/foo``) or prefix explicitly
(``openrouter/nex-agi/foo``); both route to OpenRouter.

``OPENAI_BASE_URL`` applies **only** to ``openai/...`` slugs. Roles still set to
``anthropic/...`` call Anthropic's API regardless of ``OPENAI_BASE_URL``.
``openrouter/...`` slugs use ``OPENROUTER_API_KEY`` and ``OPENROUTER_BASE_URL``.
"""
from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import Any

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
        elif provider == "openrouter":
            rps = settings.openrouter_requests_per_second
            burst = settings.openrouter_burst
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

        excs.extend([AAPIConnectionError, AAPITimeoutError, ARateLimitError])
    except Exception:
        pass
    try:
        from openai import (  # type: ignore[import-not-found]
            APIConnectionError as OAPIConnectionError,
            APIStatusError as OAPIStatusError,
            APITimeoutError as OAPITimeoutError,
            RateLimitError as ORateLimitError,
        )

        excs.extend([OAPIConnectionError, OAPITimeoutError, ORateLimitError])
    except Exception:
        pass
    return tuple(excs)


def _status_code_from_exc(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    code2 = getattr(resp, "status_code", None)
    if isinstance(code2, int):
        return code2
    return None


def _is_provider_400_error(exc: BaseException) -> bool:
    if _status_code_from_exc(exc) != 400:
        return False
    msg = str(exc).lower()
    return "provider" in msg and "error" in msg


def _should_retry_transient(exc: BaseException) -> bool:
    # Provider 400s are handled by the outer 60s retry wrapper (once).
    if _is_provider_400_error(exc):
        return False
    code = _status_code_from_exc(exc)
    if code is not None:
        return code == 429 or code >= 500
    return isinstance(exc, _transient_exceptions())


def _sleep_sync(seconds: float) -> None:
    time.sleep(seconds)


async def _sleep_async(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter (attempt is 0-based)."""
    settings = get_settings()
    base = max(0.1, settings.llm_retry_initial_delay)
    cap = max(base, settings.llm_retry_max_delay)
    ceiling = min(cap, base * (2**attempt))
    return random.uniform(0, ceiling)


def _wrap_transient_retry(runnable: Runnable) -> Runnable:
    """Retry transient LLM errors with exponential backoff (429, 5xx, network)."""
    settings = get_settings()
    max_attempts = max(1, settings.llm_max_retries)

    class _TransientRetryWrapper(Runnable):  # type: ignore[misc]
        def __init__(self, inner: Runnable):
            self._inner = inner

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002
            for attempt in range(max_attempts):
                try:
                    return self._inner.invoke(input, config=config, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if attempt + 1 >= max_attempts or not _should_retry_transient(exc):
                        raise
                    delay = _backoff_seconds(attempt)
                    _logger.warning(
                        "LLM transient error (attempt %s/%s); retrying in %.1fs: %s",
                        attempt + 1,
                        max_attempts,
                        delay,
                        str(exc)[:300],
                    )
                    _sleep_sync(delay)

        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002
            for attempt in range(max_attempts):
                try:
                    return await self._inner.ainvoke(input, config=config, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if attempt + 1 >= max_attempts or not _should_retry_transient(exc):
                        raise
                    delay = _backoff_seconds(attempt)
                    _logger.warning(
                        "LLM transient error (attempt %s/%s); retrying in %.1fs: %s",
                        attempt + 1,
                        max_attempts,
                        delay,
                        str(exc)[:300],
                    )
                    await _sleep_async(delay)

    return _TransientRetryWrapper(runnable)


def _wrap_provider_400_retry(runnable: Runnable) -> Runnable:
    """Wrap a Runnable: on provider-400 error, sleep 60s and retry once."""

    class _Provider400RetryWrapper(Runnable):  # type: ignore[misc]
        def __init__(self, inner: Runnable):
            self._inner = inner

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002
            tried = False
            while True:
                try:
                    return self._inner.invoke(input, config=config, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if tried or not _is_provider_400_error(exc):
                        raise
                    tried = True
                    _logger.warning(
                        "LLM provider 400 error; retrying once after 60s: %s",
                        str(exc)[:300],
                    )
                    _sleep_sync(60.0)

        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002
            tried = False
            while True:
                try:
                    return await self._inner.ainvoke(input, config=config, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if tried or not _is_provider_400_error(exc):
                        raise
                    tried = True
                    _logger.warning(
                        "LLM provider 400 error; retrying once after 60s: %s",
                        str(exc)[:300],
                    )
                    await _sleep_async(60.0)

    return _Provider400RetryWrapper(runnable)


def _wrap_with_retry(runnable: Runnable) -> Runnable:
    """Attach inline retry-with-backoff to a chat model or tool-bound Runnable."""
    return _wrap_transient_retry(runnable)


_KNOWN_GATEWAY_PROVIDERS = frozenset({"openai", "anthropic", "openrouter"})


def _split_slug(model_slug: str) -> tuple[str, str]:
    """Parse a role slug into (gateway_provider, model_id_for_that_gateway).

    Direct gateways: ``openai/...``, ``anthropic/...``, ``openrouter/...``.

    OpenRouter model ids are often ``vendor/model`` (``nex-agi/foo``,
    ``anthropic/claude-sonnet-4``). When the first segment is not a known
    gateway and ``OPENROUTER_API_KEY`` is set, the whole slug is sent to
    OpenRouter as the model id.
    """
    if "/" not in model_slug:
        return "openai", model_slug
    provider, name = model_slug.split("/", 1)
    provider = provider.lower()
    if provider in _KNOWN_GATEWAY_PROVIDERS:
        return provider, name
    settings = get_settings()
    if (settings.openrouter_api_key or "").strip():
        return "openrouter", model_slug
    return provider, name


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
        ("coordinator_model", settings.coordinator_model or settings.dev_model),
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
        elif provider == "openrouter":
            or_base = (settings.openrouter_base_url or "").strip().rstrip("/")
            target = f"OpenRouter {or_base} (model={model_name!r})"
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

    if provider in ("openai", "openrouter"):
        common.pop("max_tokens", None)  # ChatOpenAI uses ``max_tokens`` differently
        kwargs_openai: dict = {
            "model": name,
            "max_retries": settings.llm_max_retries,
            "timeout": settings.openai_request_timeout,
            **common,
        }
        if provider == "openai":
            kwargs_openai["api_key"] = settings.openai_api_key or None
            base = (settings.openai_base_url or "").strip()
            if base:
                kwargs_openai["base_url"] = base
                if not kwargs_openai.get("api_key"):
                    kwargs_openai["api_key"] = "not-needed"
        else:
            kwargs_openai["api_key"] = settings.openrouter_api_key or None
            or_base = (settings.openrouter_base_url or "https://openrouter.ai/api/v1").strip()
            if or_base:
                kwargs_openai["base_url"] = or_base
        return ChatOpenAI(**kwargs_openai)
    if provider == "anthropic":
        return ChatAnthropic(
            model=name,
            api_key=settings.anthropic_api_key or None,
            max_retries=settings.llm_max_retries,
            **common,
        )

    raise ValueError(
        f"Unsupported LLM provider '{provider}' in slug {model_slug!r}. "
        "Use openai/, anthropic/, or openrouter/ as the gateway prefix. "
        "For OpenRouter catalog models (e.g. nex-agi/...), set OPENROUTER_API_KEY "
        "or use openrouter/nex-agi/...."
    )


def with_retry(runnable: Runnable) -> Runnable:
    """Wrap a tool-bound model with safe retry policy.

    Policy:
      - Retry transient errors (429, 5xx, network/timeouts) with exponential backoff.
      - If we receive a "400 provider error", wait 60 seconds and retry once.
        If it happens again, raise (crash) so the failure is surfaced.
    """
    wrapped = _wrap_with_retry(runnable)
    return _wrap_provider_400_retry(wrapped)


def pm_model() -> BaseChatModel:
    s = get_settings()
    return get_chat_model(s.pm_model, temperature=s.pm_temperature)


def researcher_model() -> BaseChatModel:
    s = get_settings()
    return get_chat_model(s.researcher_model, temperature=s.researcher_temperature)


def lead_model() -> BaseChatModel:
    s = get_settings()
    return get_chat_model(s.lead_model, temperature=s.lead_temperature)


def dev_model() -> BaseChatModel:
    s = get_settings()
    return get_chat_model(s.dev_model, temperature=s.dev_temperature)


def backend_dev_model() -> BaseChatModel:
    s = get_settings()
    slug = s.backend_dev_model or s.dev_model
    return get_chat_model(slug, temperature=s.dev_temperature)


def frontend_dev_model() -> BaseChatModel:
    s = get_settings()
    slug = s.frontend_dev_model or s.dev_model
    return get_chat_model(slug, temperature=s.dev_temperature)


def devops_model() -> BaseChatModel:
    s = get_settings()
    return get_chat_model(s.devops_model, temperature=s.devops_temperature)


def qa_model() -> BaseChatModel:
    s = get_settings()
    return get_chat_model(s.qa_model, temperature=s.qa_temperature)


def coordinator_model() -> BaseChatModel:
    """Coordinator uses dev model by default (planning is done by Lead)."""
    s = get_settings()
    slug = s.coordinator_model or s.dev_model
    return get_chat_model(slug, temperature=s.coordinator_temperature)


def grader_model() -> BaseChatModel:
    s = get_settings()
    return get_chat_model(s.grader_model, temperature=s.grader_temperature)
