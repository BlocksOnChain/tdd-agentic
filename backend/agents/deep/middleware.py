"""Deep Agents middleware for tdd-agentic."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage

from backend.agents.common import emit
from backend.config import get_settings


def _preview_tool_content(content: object, limit: int = 300) -> str:
    text = content if isinstance(content, str) else str(content)
    return text if len(text) <= limit else text[:limit] + "…"


def tool_telemetry_middleware(*, agent_name: str, project_id: str):
    """Emit tool_call/tool_result events via `emit`.

    Uses LangChain's `wrap_tool_call` hook instead of streaming-parsing the graph.
    """

    @wrap_tool_call(name="ToolTelemetryMiddleware")
    async def _tool_telemetry(request: Any, handler: Callable[[Any], Awaitable[Any]]):
        tool_name = getattr(request, "name", None)
        args = getattr(request, "args", None)
        if tool_name or args:
            await emit(
                agent_name,
                "tool_call",
                {"name": tool_name, "args": args or {}},
                project_id,
            )

        result = await handler(request)

        if isinstance(result, ToolMessage):
            await emit(
                agent_name,
                "tool_result",
                {"name": result.name, "preview": _preview_tool_content(result.content)},
                project_id,
            )

        return result

    return _tool_telemetry


def _ls_provider(model: BaseChatModel) -> str | None:
    try:
        params = model._get_ls_params()  # noqa: SLF001
    except Exception:
        return None
    v = params.get("ls_provider")
    return v if isinstance(v, str) and v else None


def should_force_openai_first_tool_call(model: BaseChatModel, *, role_is_local: bool) -> bool:
    """True when this role uses OpenAI and we should force a first-hop tool call.

    OpenAI-compatible local servers often reject ``tool_choice=required``; callers
    pass ``role_is_local=True`` from settings (``*_is_local``) to skip forcing.
    """
    if role_is_local:
        return False
    return _ls_provider(model) == "openai"


def openai_first_model_call_tool_choice(request: Any) -> Any | None:
    """Return ``\"required\"`` when the next model hop should force a tool call.

    Used by deep-agent middleware for roles that use OpenAI's chat API. Returns
    ``None`` to keep the agent default (typically no forcing).
    """
    if not getattr(request, "tools", None):
        return None
    messages = getattr(request, "messages", None) or []
    if any(isinstance(m, ToolMessage) for m in messages):
        return None
    if not should_force_openai_first_tool_call(
        request.model, role_is_local=get_settings().researcher_is_local
    ):
        return None
    return "required"


def researcher_openai_first_call_tool_choice_middleware():
    """Require at least one tool call on the first model step for OpenAI chat models.

    Deep Agents otherwise leaves ``tool_choice=None``; some models answer with
    prose only (echoing assignments) and never invoke ``web_search`` / file tools.
    OpenAI-compatible local servers often reject ``tool_choice=required``; skip
    when ``RESEARCHER_IS_LOCAL`` is true (see config).
    """

    @wrap_model_call(name="ResearcherOpenAIForceFirstTool")
    async def _force_first(request: Any, handler: Callable[[Any], Awaitable[Any]]):
        choice = openai_first_model_call_tool_choice(request)
        if choice is not None:
            return await handler(request.override(tool_choice=choice))
        return await handler(request)

    return _force_first
