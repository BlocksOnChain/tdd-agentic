"""Structured logging around LLM HTTP calls — see where traffic is sent."""

from __future__ import annotations

import logging
from collections.abc import Callable

from backend.agents.llm import _split_slug  # reused slug parser
from backend.config import get_settings

_logger = logging.getLogger("backend.agents.llm_http")

_slug_for_node: dict[str, Callable[["object"], str]] = {
    "project_manager": lambda s: s.pm_model,
    "researcher": lambda s: s.researcher_model,
    "backend_lead": lambda s: s.lead_model,
    "frontend_lead": lambda s: s.lead_model,
    "backend_dev": lambda s: s.backend_dev_model or s.dev_model,
    "frontend_dev": lambda s: s.frontend_dev_model or s.dev_model,
    "devops": lambda s: s.dev_model,
    "qa": lambda s: s.dev_model,
}


def describe_llm_slug(slug: str) -> tuple[str, str, str]:
    """Return (provider, model_name, human_readable_endpoint_hint)."""
    settings = get_settings()
    provider, model_name = _split_slug(slug)
    if provider == "openai":
        base = (settings.openai_base_url or "").strip().rstrip("/")
        if base:
            completions = f"{base}/chat/completions"
            return provider, model_name, f"OpenAI-compatible POST {completions}"
        return provider, model_name, "OpenAI platform API (chat.completions)"
    return provider, model_name, "Anthropic Messages API"


def resolve_model_slug_for_node(node_name: str) -> str:
    settings = get_settings()
    getter = _slug_for_node.get(node_name)
    if getter is None:
        return "?/?"
    return getter(settings)


def log_llm_invoke_start(
    *,
    node_name: str,
    step_index: int,
    step_cap: int,
    project_id: str | None,
) -> None:
    if not get_settings().llm_invoke_log_each_call:
        return
    slug = resolve_model_slug_for_node(node_name)
    provider, model_name, endpoint = describe_llm_slug(slug)
    pid = project_id or "(unknown)"
    _logger.info(
        "LLM call start node=%s project_id=%s step=%s/%s slug=%s provider=%s model=%r endpoint=%s",
        node_name,
        pid[:8] + ("…" if len(pid) > 8 else ""),
        step_index,
        step_cap,
        slug,
        provider,
        model_name,
        endpoint,
    )


def log_rag_crag_llm_targets(*, docs_to_grade: int, grader_slug: str, rewriter_slug: str) -> None:
    if not get_settings().llm_invoke_log_each_call:
        return
    _, _, g_ep = describe_llm_slug(grader_slug)
    _, _, r_ep = describe_llm_slug(rewriter_slug)
    _logger.info(
        "RAG CRAG LLM routes: grade_batch docs=%s grader=%s target=%s | rewriter=%s target=%s",
        docs_to_grade,
        grader_slug,
        g_ep,
        rewriter_slug,
        r_ep,
    )


def log_llm_invoke_exception_context(
    *,
    node_name: str,
    step_index: int,
    project_id: str | None,
) -> None:
    """Call from inside an ``except`` block — attaches stack trace via ``exception()``."""
    slug = resolve_model_slug_for_node(node_name)
    _, _, endpoint = describe_llm_slug(slug)
    pid = project_id or "(unknown)"
    _logger.exception(
        "LLM call FAILED node=%s project_id=%s step=%s slug=%s target=%s",
        node_name,
        pid,
        step_index,
        slug,
        endpoint,
    )
