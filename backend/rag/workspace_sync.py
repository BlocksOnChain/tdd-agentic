"""Sync researcher-authored workspace markdown into the project RAG index."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.agents.researcher.scaffold import should_skip_research_rag_index
from backend.config import get_settings
from backend.rag.ingestion import ingest_text


def project_workspace_root(project_id: str) -> Path:
    root = (get_settings().workspace_root / project_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def iter_research_markdown_paths(project_id: str) -> list[Path]:
    """Markdown files the researcher is expected to author for retrieval."""
    root = project_workspace_root(project_id)
    paths: list[Path] = []
    agents = root / "AGENTS.md"
    if agents.is_file():
        paths.append(agents)
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        paths.extend(sorted(p for p in docs_dir.rglob("*.md") if p.is_file()))
    skills_dir = root / "_skills"
    if skills_dir.is_dir():
        paths.extend(sorted(p for p in skills_dir.rglob("SKILL.md") if p.is_file()))
    return paths


def list_research_markdown(project_id: str) -> list[dict[str, Any]]:
    root = project_workspace_root(project_id)
    rows: list[dict[str, Any]] = []
    for path in iter_research_markdown_paths(project_id):
        rel = path.relative_to(root).as_posix()
        rows.append({"path": rel, "bytes": path.stat().st_size})
    return rows


async def ingest_research_workspace(project_id: str) -> dict[str, Any]:
    """Embed every researcher markdown file under the project workspace."""
    root = project_workspace_root(project_id)
    files: list[dict[str, Any]] = []
    total_chunks = 0
    for path in iter_research_markdown_paths(project_id):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
            if should_skip_research_rag_index(project_id, rel, text):
                files.append({"path": rel, "skipped": "scaffold_placeholder"})
                continue
            chunks = await ingest_text(project_id, text, source=rel)
            total_chunks += chunks
            files.append({"path": rel, "chunks_written": chunks, "bytes": len(text.encode("utf-8"))})
        except Exception as exc:  # noqa: BLE001
            files.append({"path": rel, "error": str(exc)})
    return {
        "files": files,
        "file_count": len(files),
        "total_chunks": total_chunks,
    }
