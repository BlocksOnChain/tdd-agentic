"""Human-readable summaries for agent activity rows."""
from __future__ import annotations

import json
from typing import Any

_MAX_DETAIL_CHARS = 2000


def _truncate(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def format_agent_log_detail(kind: str, payload: dict[str, Any]) -> str:
    """Render a compact activity line for dashboards and log replay."""
    if not payload:
        return kind

    if kind == "tool_result":
        name = payload.get("name") or "tool"
        preview = payload.get("preview") or ""
        return _truncate(f"{name}: {preview}")

    if kind == "research_artifacts":
        tools = payload.get("tools") if isinstance(payload.get("tools"), dict) else {}
        tool_calls = tools.get("tool_calls") if isinstance(tools, dict) else {}
        sync = payload.get("workspace_sync") if isinstance(payload.get("workspace_sync"), dict) else {}
        files = payload.get("workspace_files") if isinstance(payload.get("workspace_files"), list) else []
        paths = ", ".join(
            str(item.get("path"))
            for item in files
            if isinstance(item, dict) and item.get("path")
        )
        return _truncate(
            "research artifacts · "
            f"tool_calls={tool_calls or {}} · "
            f"rag_ingest_chunks={tools.get('rag_ingest_chunks', 0)} · "
            f"workspace_files={len(files)} · "
            f"indexed_chunks={sync.get('total_chunks', 0)}"
            + (f" · paths={paths}" if paths else "")
        )

    if kind in {"route", "route_fallback"}:
        target = payload.get("next_agent") or "?"
        rationale = payload.get("rationale") or ""
        return _truncate(f"→ {target}: {rationale}")

    if kind == "turn_end":
        if not payload:
            return "turn complete"
        return _truncate(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    if kind == "turn_start":
        backend = payload.get("web_search_backend")
        if isinstance(backend, str) and backend.strip():
            return _truncate(f"turn started · web_search={backend}")
        return "turn started"

    if kind == "tool_call":
        name = payload.get("name") or payload.get("tool") or "tool"
        preview = payload.get("preview") or payload.get("args") or ""
        return _truncate(f"{name}: {preview}")

    for key in ("preview", "rationale", "error", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(value)

    if kind == "log" and isinstance(payload.get("update"), dict):
        return _truncate(json.dumps(payload.get("update"), ensure_ascii=False)[:1200])

    return _truncate(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
