"""Researcher workspace markdown discovery and RAG sync."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.researcher import finalize as researcher_finalize
from backend.agents.researcher import scaffold as researcher_scaffold
from backend.rag import workspace_sync


@pytest.mark.asyncio
async def test_ingest_research_workspace_skips_scaffold_placeholders(
    monkeypatch, tmp_path: Path
) -> None:
    project_id = "proj-scaffold-rag"
    root = tmp_path / project_id
    (root / "docs").mkdir(parents=True)
    monkeypatch.setattr(workspace_sync, "project_workspace_root", lambda pid: root)
    monkeypatch.setattr(researcher_finalize, "project_workspace_root", lambda pid: root)
    monkeypatch.setattr(researcher_scaffold, "project_workspace_root", lambda pid: root)
    researcher_finalize.ensure_research_docs_scaffold(project_id, "Todo app with Next.js and Express")

    calls: list[str] = []

    async def fake_ingest(pid: str, text: str, *, source: str) -> int:
        calls.append(source)
        return 1

    monkeypatch.setattr(workspace_sync, "ingest_text", fake_ingest)

    report = await workspace_sync.ingest_research_workspace(project_id)
    assert calls == []
    skipped = [row for row in report["files"] if row.get("skipped") == "scaffold_placeholder"]
    assert len(skipped) == 2


@pytest.mark.asyncio
async def test_ingest_research_workspace_indexes_docs(monkeypatch, tmp_path: Path) -> None:
    project_id = "proj-research"
    root = tmp_path / project_id
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "tech-stack.md").write_text("# Stack\n\nNext.js + Prisma", encoding="utf-8")
    (root / "AGENTS.md").write_text("# Rules", encoding="utf-8")

    monkeypatch.setattr(workspace_sync, "project_workspace_root", lambda pid: root)

    calls: list[tuple[str, str, str]] = []

    async def fake_ingest(pid: str, text: str, *, source: str) -> int:
        calls.append((pid, source, text))
        return max(1, len(text) // 40)

    monkeypatch.setattr(workspace_sync, "ingest_text", fake_ingest)

    report = await workspace_sync.ingest_research_workspace(project_id)
    assert report["file_count"] == 2
    assert report["total_chunks"] >= 2
    sources = {source for _, source, _ in calls}
    assert sources == {"AGENTS.md", "docs/tech-stack.md"}


def test_refresh_authored_scaffold_paths_clears_rewritten_docs(monkeypatch, tmp_path: Path) -> None:
    project_id = "proj-scaffold-refresh"
    root = tmp_path / project_id
    (root / "docs").mkdir(parents=True)
    monkeypatch.setattr(workspace_sync, "project_workspace_root", lambda pid: root)
    monkeypatch.setattr(researcher_finalize, "project_workspace_root", lambda pid: root)
    monkeypatch.setattr(researcher_scaffold, "project_workspace_root", lambda pid: root)

    researcher_finalize.ensure_research_docs_scaffold(project_id, "Todo app with Next.js and Express")
    assert researcher_finalize.substantive_research_docs_present(project_id) is False

    tech_stack = root / "docs" / "tech-stack.md"
    tech_stack.write_text(
        "# Tech stack\n\n"
        "Frontend: Next.js 15 with React 19.\n"
        "Backend: Express 5 with TypeScript.\n"
        "Database: PostgreSQL via Prisma.\n"
        "Testing: Vitest for frontend, pytest for backend.\n",
        encoding="utf-8",
    )
    cleared = researcher_scaffold.refresh_authored_scaffold_paths(project_id)
    assert cleared == ["docs/tech-stack.md"]
    assert researcher_finalize.substantive_research_docs_present(project_id) is True


def test_scaffolded_docs_do_not_count_as_substantive(monkeypatch, tmp_path: Path) -> None:
    project_id = "proj-scaffold-gate"
    root = tmp_path / project_id
    (root / "docs").mkdir(parents=True)
    monkeypatch.setattr(workspace_sync, "project_workspace_root", lambda pid: root)
    monkeypatch.setattr(researcher_finalize, "project_workspace_root", lambda pid: root)
    monkeypatch.setattr(researcher_scaffold, "project_workspace_root", lambda pid: root)

    researcher_finalize.ensure_research_docs_scaffold(project_id, "Todo app with Next.js and Express")
    assert researcher_finalize.substantive_research_docs_present(project_id) is False


def test_ensure_research_docs_scaffold_writes_missing_docs(monkeypatch, tmp_path: Path) -> None:
    project_id = "proj-scaffold"
    root = tmp_path / project_id
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "README.md").write_text("# README", encoding="utf-8")
    monkeypatch.setattr(workspace_sync, "project_workspace_root", lambda pid: root)
    monkeypatch.setattr(researcher_finalize, "project_workspace_root", lambda pid: root)
    monkeypatch.setattr(researcher_scaffold, "project_workspace_root", lambda pid: root)

    written = researcher_finalize.ensure_research_docs_scaffold(
        project_id,
        "Todo app with Next.js and Express",
    )
    assert written == ["docs/tech-stack.md", "docs/architecture.md"]
    assert (root / "docs" / "tech-stack.md").is_file()
    assert "Next.js" in (root / "docs" / "tech-stack.md").read_text(encoding="utf-8")


def test_list_research_markdown_reports_workspace_files(monkeypatch, tmp_path: Path) -> None:
    project_id = "proj-list"
    root = tmp_path / project_id
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "architecture.md").write_text("# Architecture", encoding="utf-8")
    monkeypatch.setattr(workspace_sync, "project_workspace_root", lambda pid: root)

    rows = workspace_sync.list_research_markdown(project_id)
    assert rows == [{"path": "docs/architecture.md", "bytes": len("# Architecture")}]
