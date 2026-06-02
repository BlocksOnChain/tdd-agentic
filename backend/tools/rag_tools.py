"""LangChain tools for RAG ingestion and CRAG-style retrieval."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from backend.rag import ingestion, retrieval


@tool
async def rag_ingest_text(project_id: str, source: str, text: str) -> str:
    """Add a document to the project's RAG vector store.

    USE WHEN: You wrote documentation, research findings, or spec that other agents will need.
    AVOID WHEN: The text already exists in RAG — ingestion is idempotent on (source, chunk) so no harm in re-ingesting.
    AVOID WHEN: The text is trivial (one line) — keep chunks above ~50 chars for retrieval quality.

    ``source`` should be a stable identifier (e.g. file path or URL).
    Returns: {"chunks_written": N, "source": "...", "project_id": "..."}
    """
    n = await ingestion.ingest_text(project_id, text, source=source)
    return json.dumps({"chunks_written": n, "source": source, "project_id": project_id})


@tool
async def rag_query(project_id: str, query: str, k: int | None = None) -> str:
    """Retrieve relevant project documents using the CRAG pipeline.

    USE WHEN: You need context about libraries, patterns, or decisions from this project.
    AVOID WHEN: You need ticket state — use list_tickets/get_ticket for that.
    AVOID WHEN: You need full SKILL.md content — call the skill directly if you know the name.

    CRAG pipeline: vector search → relevance grading → query rewrite on total miss.

    Common queries:
      - "auth patterns used in this project" → finds auth-related docs
      - "testing setup" → finds test configuration and framework docs
      - "tech stack" → finds architecture and library docs

    Returns: Context block with up to k documents, each tagged [source: <path>].
    On no results: "(no relevant documents found)"
    """
    from backend.config import get_settings

    limit = k if k is not None else get_settings().rag_default_k
    docs = await retrieval.crag_retrieve(project_id, query, k=limit)
    return await retrieval.format_context(docs)
