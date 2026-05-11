"""Researcher turn finalization: tool visibility and workspace → RAG sync."""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from backend.agents.common import emit
from backend.agents.researcher.scaffold import (
    is_scaffolded_path,
    load_scaffolded_paths,
    record_scaffolded_paths,
    refresh_authored_scaffold_paths,
)
from backend.config import get_settings
from backend.rag.workspace_sync import (
    ingest_research_workspace,
    list_research_markdown,
    project_workspace_root,
)


def _preview_tool_content(content: object, limit: int = 300) -> str:
    text = content if isinstance(content, str) else str(content)
    return text if len(text) <= limit else text[:limit] + "…"


_CONTEXT_STOPWORDS = frozenset(
    {
        "with",
        "that",
        "this",
        "from",
        "have",
        "will",
        "good",
        "want",
        "your",
        "their",
        "more",
        "some",
        "into",
        "when",
        "what",
        "also",
        "been",
        "than",
        "them",
        "they",
        "only",
        "very",
        "just",
        "like",
        "make",
        "need",
        "work",
        "using",
        "used",
        "build",
        "list",
    }
)

_OFFTOPIC_SUMMARY_PATTERNS = (
    re.compile(r"\b(?:summer internship|research assistant position)\b", re.I),
    re.compile(r"\b(?:3rd|third)\s+year\s+student\b", re.I),
    re.compile(r"\b(?:curriculum vitae|personal bio|cover letter)\b", re.I),
    re.compile(r"\b(?:university in the uk|bsc in computer science)\b", re.I),
)


def _summary_looks_offtopic(summary: str) -> bool:
    return any(pattern.search(summary) for pattern in _OFFTOPIC_SUMMARY_PATTERNS)


def _summary_relates_to_project(summary: str, project_context: str) -> bool:
    cleaned = summary.strip()
    if not cleaned:
        return False
    if _summary_looks_offtopic(cleaned):
        return False

    blob = cleaned.lower()
    if "docs/" in blob or ".md" in blob:
        return True

    ctx = project_context.strip().lower()
    if not ctx:
        return False

    tokens = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", ctx)
        if token not in _CONTEXT_STOPWORDS
    }
    if not tokens:
        return False
    return sum(1 for token in tokens if token in blob) >= 2


def build_researcher_handoff_summary(
    *,
    ai_text: str,
    project_context: str,
    turn_end_payload: dict[str, Any],
) -> str:
    paths = [
        str(row.get("path") or "")
        for row in turn_end_payload.get("workspace_files") or []
        if row.get("path")
    ]
    tools = turn_end_payload.get("tools") or {}
    rag_chunks = int(tools.get("rag_ingest_chunks") or 0)
    incomplete = bool(turn_end_payload.get("research_incomplete"))
    if incomplete:
        structured = (
            "Research incomplete for the current project goal. "
            "Substantive docs/ files are still missing. "
            f"Workspace markdown present: {', '.join(paths) if paths else 'none'}. "
            f"RAG chunks ingested via tools this turn: {rag_chunks}."
        )
    else:
        structured = (
            "Research artifacts for the current project goal. "
            f"Workspace docs: {', '.join(paths) if paths else 'none yet'}. "
            f"RAG chunks ingested via tools this turn: {rag_chunks}."
        )

    cleaned = (ai_text or "").strip()
    if not incomplete and cleaned and _summary_relates_to_project(cleaned, project_context):
        return cleaned
    return structured


RESEARCHER_CONTINUATION_NUDGE = (
    "The previous research pass did not produce substantive project documentation. "
    "Call web_search for the project's stack and deliverables, write or refresh "
    "docs/tech-stack.md, docs/architecture.md, docs/conventions.md, and "
    "docs/api-contracts.md with project-specific content using write_file or "
    "edit_file, and call rag_ingest_text for each authored file before finishing. "
    "Auto-scaffolded placeholders do not satisfy project_manager routing."
)


def build_researcher_deep_assignment_messages(project_id: str) -> list[HumanMessage]:
    """Warn the deep researcher about placeholder docs that must be replaced."""
    refresh_authored_scaffold_paths(project_id)
    if substantive_research_docs_present(project_id):
        return []

    scaffolded = sorted(load_scaffolded_paths(project_id))
    if scaffolded:
        paths = ", ".join(scaffolded)
        return [
            HumanMessage(
                content=(
                    "[research workspace status]\n"
                    f"Placeholder docs still need substantive rewrites: {paths}. "
                    "Use web_search, then write_file or edit_file to replace them, "
                    "then rag_ingest_text for each authored file."
                )
            )
        ]
    return [
        HumanMessage(
            content=(
                "[research workspace status]\n"
                "No substantive docs/ research artifacts exist yet. "
                "Use web_search, write_file or edit_file under docs/, and "
                "rag_ingest_text before finishing."
            )
        )
    ]


def substantive_research_docs_present(project_id: str) -> bool:
    refresh_authored_scaffold_paths(project_id)
    for row in list_research_markdown(project_id):
        path = str(row.get("path") or "")
        if int(row.get("bytes") or 0) <= 0:
            continue
        if not path.startswith("docs/") or not path.endswith(".md"):
            continue
        if path == "docs/README.md":
            continue
        if is_scaffolded_path(project_id, path):
            continue
        return True
    return False


def researcher_turn_incomplete(project_id: str, tool_msgs: list[ToolMessage]) -> bool:
    """True when the researcher still owes substantive docs/ for PM routing."""
    del tool_msgs
    refresh_authored_scaffold_paths(project_id)
    return not substantive_research_docs_present(project_id)


def ensure_research_docs_scaffold(project_id: str, project_context: str) -> list[str]:
    """Write minimal docs when the model ends a turn without substantive docs/."""
    if substantive_research_docs_present(project_id):
        return []

    root = project_workspace_root(project_id)
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    brief = project_context.strip() or "No project brief was provided."
    templates = {
        "docs/tech-stack.md": (
            "# Tech stack\n\n"
            f"Project brief:\n\n{brief}\n\n"
            "Document the chosen frontend, backend, database, and test tooling here.\n"
        ),
        "docs/architecture.md": (
            "# Architecture\n\n"
            f"Project brief:\n\n{brief}\n\n"
            "Describe the main components, data flow, and deployment shape here.\n"
        ),
    }
    written: list[str] = []
    for rel_path, content in templates.items():
        path = root / rel_path
        if path.is_file() and path.stat().st_size > 0:
            continue
        path.write_text(content, encoding="utf-8")
        written.append(rel_path)
    record_scaffolded_paths(project_id, written)
    return written


def summarize_researcher_tools(tool_msgs: list[ToolMessage]) -> dict[str, Any]:
    counts = Counter(tm.name for tm in tool_msgs if tm.name)
    rag_chunks = 0
    for tm in tool_msgs:
        if tm.name != "rag_ingest_text":
            continue
        try:
            payload = json.loads(str(tm.content))
        except json.JSONDecodeError:
            continue
        rag_chunks += int(payload.get("chunks_written") or 0)
    return {
        "tool_calls": dict(counts),
        "rag_ingest_chunks": rag_chunks,
    }


async def finalize_researcher_turn(
    *,
    agent: str,
    project_id: str,
    tool_msgs: list[ToolMessage],
    emit_tool_results: bool,
    project_context: str = "",
) -> dict[str, Any]:
    if emit_tool_results:
        for tm in tool_msgs:
            await emit(
                agent,
                "tool_result",
                {"name": tm.name, "preview": _preview_tool_content(tm.content)},
                project_id,
            )

    tools = summarize_researcher_tools(tool_msgs)
    refresh_authored_scaffold_paths(project_id)
    scaffolded: list[str] = []
    if not get_settings().use_deep_agent_researcher:
        scaffolded = ensure_research_docs_scaffold(project_id, project_context)
        refresh_authored_scaffold_paths(project_id)
    workspace_sync = await ingest_research_workspace(project_id)
    workspace_files = list_research_markdown(project_id)
    payload = {
        "tools": tools,
        "workspace_files": workspace_files,
        "workspace_sync": workspace_sync,
        "scaffolded_docs": scaffolded,
        "research_incomplete": not substantive_research_docs_present(project_id),
    }
    await emit(agent, "research_artifacts", payload, project_id)
    return payload
