"""LangChain tools for RAG ingestion and CRAG-style retrieval."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from backend.rag import ingestion, retrieval


@tool
async def rag_ingest_text(project_id: str, source: str, text: str) -> str:
    """Add a document to the project's RAG vector store.

    ``source`` should be a stable identifier (e.g. file path or URL).
    Returns the number of chunks written.
    """
    from backend.agents.researcher.scaffold import (
        normalize_workspace_relative_path,
        should_skip_research_rag_index,
    )

    rel = normalize_workspace_relative_path(source)
    looks_like_workspace_file = bool(rel) and (
        "://" not in source
        and not rel.lower().startswith("skill:")
        and (rel.lower().startswith("docs/") or rel.lower() == "agents.md")
    )
    if looks_like_workspace_file and should_skip_research_rag_index(project_id, rel, text):
        return json.dumps(
            {
                "chunks_written": 0,
                "source": source,
                "project_id": project_id,
                "skipped": "placeholder_or_non_research_workspace_doc",
            }
        )

    n = await ingestion.ingest_text(project_id, text, source=source)
    return json.dumps({"chunks_written": n, "source": source, "project_id": project_id})


@tool
async def rag_query(project_id: str, query: str, k: int | None = None) -> str:
    """Retrieve relevant project documents using the CRAG pipeline."""
    from backend.config import get_settings

    limit = k if k is not None else get_settings().rag_default_k
    docs = await retrieval.crag_retrieve(project_id, query, k=limit)
    return await retrieval.format_context(docs)
