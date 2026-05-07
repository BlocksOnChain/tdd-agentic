"""CRAG-style retrieval pipeline: retrieve → grade → (rewrite | return).

Implemented directly against ``AsyncQdrantClient`` for full async control.
A small grader LLM filters out irrelevant chunks. If nothing relevant
survives, the query is rewritten and we retry once before giving up.
"""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from backend.agents.llm import grader_model, researcher_model
from backend.config import get_settings
from backend.rag.embeddings import get_embeddings
from backend.rag.ingestion import PAYLOAD_TEXT_KEY, collection_name, ensure_collection


class _RelevanceScore(BaseModel):
    is_relevant: bool = Field(description="True if document directly answers the query")


_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether a retrieved document is relevant to a user query. "
            "Reply with a JSON object {is_relevant: true/false}. Be strict — only mark "
            "relevant if the document contains information that helps answer the query.",
        ),
        ("human", "Query:\n{query}\n\nDocument:\n{document}"),
    ]
)


_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite search queries to retrieve more relevant documents. "
            "Produce a single, focused query optimized for vector search.",
        ),
        ("human", "Original query: {query}\nReturn ONLY the rewritten query."),
    ]
)


def _point_to_document(payload: dict[str, Any] | None, score: float | None) -> Document:
    payload = dict(payload or {})
    text = str(payload.pop(PAYLOAD_TEXT_KEY, ""))
    if score is not None:
        payload["_score"] = score
    return Document(page_content=text, metadata=payload)


async def _vector_search(project_id: str, query: str, k: int) -> list[Document]:
    await ensure_collection(project_id)
    settings = get_settings()
    embeddings = get_embeddings()
    qvec = await embeddings.aembed_query(query)

    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        # qdrant-client 1.10+ uses query_points; .search was removed.
        if hasattr(client, "query_points"):
            result = await client.query_points(
                collection_name=collection_name(project_id),
                query=qvec,
                limit=k,
                with_payload=True,
            )
            hits = getattr(result, "points", []) or []
        else:
            hits = await client.search(  # type: ignore[attr-defined]
                collection_name=collection_name(project_id),
                query_vector=qvec,
                limit=k,
                with_payload=True,
            )
        return [_point_to_document(h.payload, getattr(h, "score", None)) for h in hits]
    finally:
        await client.close()


async def _grade(query: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return []
    grader = grader_model().with_structured_output(_RelevanceScore)
    kept: list[Document] = []
    for doc in docs:
        try:
            score = await grader.ainvoke(
                _GRADER_PROMPT.format_messages(
                    query=query, document=doc.page_content[:2000]
                )
            )
            if score.is_relevant:
                kept.append(doc)
        except Exception:
            kept.append(doc)
    return kept


async def _rewrite(query: str) -> str:
    rewriter = researcher_model()
    msg = await rewriter.ainvoke(_REWRITE_PROMPT.format_messages(query=query))
    return str(msg.content).strip().strip('"')


async def crag_retrieve(project_id: str, query: str, k: int = 6) -> list[Document]:
    """Retrieve and grade. On total miss, rewrite once and retry."""
    initial = await _vector_search(project_id, query, k)
    relevant = await _grade(query, initial)
    if relevant:
        return relevant

    rewritten = await _rewrite(query)
    second = await _vector_search(project_id, rewritten, k)
    return await _grade(rewritten, second)


async def format_context(docs: list[Document], max_chars: int = 6000) -> str:
    """Render documents as a single context block suitable for an LLM prompt."""
    if not docs:
        return "(no relevant documents found)"
    parts: list[str] = []
    used = 0
    for d in docs:
        chunk = f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n---\n\n".join(parts)
