"""Deep-agent tool wiring."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import create_model


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


__all__ = ["bind_tools_to_project"]
