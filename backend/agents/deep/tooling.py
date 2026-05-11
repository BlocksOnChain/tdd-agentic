"""Deep-agent tool wiring and activity telemetry."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import create_model
from backend.agents.common import emit


def _tool_field_names(tool: BaseTool) -> set[str]:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return set()
    fields = getattr(schema, "model_fields", None)
    if isinstance(fields, Mapping):
        return {str(key) for key in fields}
    try:
        return set(schema.schema().get("properties", {}))
    except Exception:  # noqa: BLE001
        return set()


def _args_schema_without_project_id(schema: type[Any] | None) -> type[Any] | None:
    if schema is None or not hasattr(schema, "model_fields"):
        return None
    field_defs = {
        name: (field.annotation, field)
        for name, field in schema.model_fields.items()
        if name != "project_id"
    }
    if not field_defs:
        return None
    return create_model(f"{schema.__name__}ProjectBound", **field_defs)


def _bind_single_tool_to_project(tool: BaseTool, project_id: str) -> BaseTool:
    async def _invoke(**kwargs: Any) -> Any:
        payload = dict(kwargs)
        payload["project_id"] = project_id
        if tool.coroutine is not None:
            return await tool.coroutine(**payload)
        return await tool.ainvoke(payload)

    description = (tool.description or "").strip()
    if description:
        description = f"{description} (project_id is injected automatically.)"
    return StructuredTool.from_function(
        coroutine=_invoke,
        name=tool.name,
        description=description or None,
        args_schema=_args_schema_without_project_id(tool.args_schema),
    )


def bind_tools_to_project(tools: Sequence[BaseTool], project_id: str) -> list[BaseTool]:
    """Bind project-scoped tools so the model does not need to pass ``project_id``."""
    bound: list[BaseTool] = []
    for tool in tools:
        if "project_id" not in _tool_field_names(tool):
            bound.append(tool)
            continue
        bound.append(_bind_single_tool_to_project(tool, project_id))
    return bound


def _preview_tool_content(content: object, limit: int = 300) -> str:
    text = content if isinstance(content, str) else str(content)
    return text if len(text) <= limit else text[:limit] + "…"


def _iter_stream_messages(update: object) -> list[BaseMessage]:
    if not isinstance(update, dict):
        return []
    messages = update.get("messages")
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, BaseMessage)]


async def invoke_deep_agent_with_tool_telemetry(
    agent: Any,
    *,
    agent_name: str,
    project_id: str,
    input_state: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[ToolMessage]]:
    """Run a deep agent, emitting tool activity and collecting tool results."""
    tool_messages: list[ToolMessage] = []
    seen_tool_results: set[str] = set()
    final_state: dict[str, Any] = {}

    async for mode, chunk in agent.astream(
        input_state,
        config=config,
        stream_mode=["values", "updates"],
    ):
        if mode == "values" and isinstance(chunk, dict):
            final_state = chunk
            continue
        if mode != "updates" or not isinstance(chunk, dict):
            continue
        for update in chunk.values():
            for message in _iter_stream_messages(update):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls or []:
                        await emit(
                            agent_name,
                            "tool_call",
                            {
                                "name": call.get("name"),
                                "args": call.get("args") or {},
                            },
                            project_id,
                        )
                if not isinstance(message, ToolMessage):
                    continue
                tool_call_id = message.tool_call_id or ""
                if tool_call_id and tool_call_id in seen_tool_results:
                    continue
                if tool_call_id:
                    seen_tool_results.add(tool_call_id)
                tool_messages.append(message)
                await emit(
                    agent_name,
                    "tool_result",
                    {
                        "name": message.name,
                        "preview": _preview_tool_content(message.content),
                    },
                    project_id,
                )

    if not final_state:
        raise RuntimeError(
            f"Deep agent {agent_name!r} finished without emitting a final state snapshot."
        )
    return final_state, tool_messages
