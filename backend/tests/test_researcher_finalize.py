"""Researcher handoff summary helpers."""
from __future__ import annotations

from backend.agents.researcher.finalize import (
    build_researcher_deep_assignment_messages,
    build_researcher_handoff_summary,
    researcher_turn_incomplete,
)


def test_researcher_handoff_replaces_unrelated_model_summary() -> None:
    summary = build_researcher_handoff_summary(
        ai_text="I am a 3rd year student studying computer science in the UK.",
        project_context="Build a todo app with shadcn, Prisma, and PostgreSQL.",
        turn_end_payload={
            "workspace_files": [{"path": "docs/tech-stack.md", "bytes": 120}],
            "tools": {"rag_ingest_chunks": 2},
        },
    )
    assert "student" not in summary.lower()
    assert "docs/tech-stack.md" in summary


def test_researcher_handoff_keeps_on_topic_model_summary() -> None:
    summary = build_researcher_handoff_summary(
        ai_text="Documented shadcn/ui and Prisma in docs/tech-stack.md and ingested into RAG.",
        project_context="Build a todo app with shadcn, Prisma, and PostgreSQL.",
        turn_end_payload={
            "workspace_files": [{"path": "docs/tech-stack.md", "bytes": 120}],
            "tools": {"rag_ingest_chunks": 1},
        },
    )
    assert "docs/tech-stack.md" in summary


def test_researcher_handoff_replaces_resume_style_summary() -> None:
    summary = build_researcher_handoff_summary(
        ai_text=(
            "I am a 3rd year student at a university in the UK studying a BSc in Computer Science. "
            "I am looking for a summer internship or a research assistant position."
        ),
        project_context="I want a todo list with good quality UI use next js and express (nodejs), prism for database",
        turn_end_payload={
            "workspace_files": [{"path": "docs/README.md", "bytes": 120}],
            "tools": {"rag_ingest_chunks": 0},
        },
    )
    assert "internship" not in summary.lower()
    assert "Research artifacts for the current project goal." in summary


def test_researcher_deep_assignment_flags_scaffolded_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agents.researcher.finalize.substantive_research_docs_present",
        lambda _pid: False,
    )
    monkeypatch.setattr(
        "backend.agents.researcher.finalize.refresh_authored_scaffold_paths",
        lambda _pid: [],
    )
    monkeypatch.setattr(
        "backend.agents.researcher.finalize.load_scaffolded_paths",
        lambda _pid: {"docs/tech-stack.md"},
    )
    messages = build_researcher_deep_assignment_messages("proj-1")
    assert len(messages) == 1
    assert "docs/tech-stack.md" in str(messages[0].content)


def test_researcher_turn_incomplete_without_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agents.researcher.finalize.substantive_research_docs_present",
        lambda _pid: False,
    )
    monkeypatch.setattr(
        "backend.agents.researcher.finalize.refresh_authored_scaffold_paths",
        lambda _pid: [],
    )
    assert researcher_turn_incomplete("proj-1", []) is True


def test_researcher_handoff_rejects_single_stopword_overlap() -> None:
    summary = build_researcher_handoff_summary(
        ai_text="I am a good communicator and enjoy working collaboratively with others.",
        project_context="I want a todo list with good quality UI use next js and express (nodejs), prism for database",
        turn_end_payload={
            "workspace_files": [{"path": "docs/README.md", "bytes": 120}],
            "tools": {"rag_ingest_chunks": 0},
        },
    )
    assert "communicator" not in summary.lower()
    assert "Research artifacts for the current project goal." in summary
