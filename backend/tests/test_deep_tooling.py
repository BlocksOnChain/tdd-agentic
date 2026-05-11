"""Deep-agent tool wiring helpers."""
from __future__ import annotations

import json

import pytest
from langchain_core.tools import tool

from backend.agents.deep.tooling import bind_tools_to_project


@tool
async def rag_ingest_text(project_id: str, source: str, text: str) -> str:
  """Ingest text into RAG."""
  return json.dumps({"project_id": project_id, "source": source, "bytes": len(text)})


@pytest.mark.asyncio
async def test_bind_tools_to_project_injects_project_id() -> None:
    bound = bind_tools_to_project([rag_ingest_text], "proj-123")
    assert len(bound) == 1
    raw = await bound[0].ainvoke({"source": "docs/tech-stack.md", "text": "hello"})
    payload = json.loads(raw)
    assert payload["project_id"] == "proj-123"
    assert payload["source"] == "docs/tech-stack.md"
