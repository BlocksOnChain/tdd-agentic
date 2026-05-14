
"""Optional Langfuse callback handler for LangGraph traces.

Returns ``None`` when Langfuse credentials aren't configured so callers can
unconditionally include the result in a ``callbacks=[...]`` list.
"""
from __future__ import annotations

from functools import lru_cache

from backend.config import get_settings


@lru_cache(maxsize=1)
def get_langfuse_handler():
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        return None


def callbacks_for(project_id: str | None = None) -> list:
    handler = get_langfuse_handler()
    if handler is None:
        return []
    if project_id is not None:
        try:
            handler.session_id = project_id  # best-effort thread tagging
        except Exception:
            pass
    return [handler]
