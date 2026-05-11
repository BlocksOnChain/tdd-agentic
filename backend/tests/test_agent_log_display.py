"""Agent activity detail formatting."""
from __future__ import annotations

from backend.agent_logs.display import format_agent_log_detail


def test_format_tool_result_detail() -> None:
    detail = format_agent_log_detail(
        "tool_result",
        {"name": "rag_ingest_text", "preview": '{"chunks_written": 3}'},
    )
    assert "rag_ingest_text" in detail
    assert "chunks_written" in detail


def test_format_research_artifacts_detail() -> None:
    detail = format_agent_log_detail(
        "research_artifacts",
        {
            "tools": {"tool_calls": {"rag_ingest_text": 2}, "rag_ingest_chunks": 5},
            "workspace_files": [{"path": "docs/tech-stack.md", "bytes": 120}],
            "workspace_sync": {"total_chunks": 5, "file_count": 1},
        },
    )
    assert "research artifacts" in detail
    assert "docs/tech-stack.md" in detail
    assert "indexed_chunks=5" in detail
