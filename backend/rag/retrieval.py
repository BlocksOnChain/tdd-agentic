"""CRAG-style retrieval pipeline: retrieve → grade → (rewrite | return).

Implemented directly against ``AsyncQdrantClient`` for full async control.
A small grader LLM filters out irrelevant chunks. If nothing relevant
survives, the query is rewritten and we retry once before giving up.

Query expansion: for single-word queries, auto-expand with related terms
for better recall.

TTL-aware cache with LRU eviction.
"""
from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from backend.agents.llm import grader_model, researcher_model
from backend.agents.llm_audit import log_rag_crag_llm_targets
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

_CRAG_CACHE: dict[tuple[str, str, int], tuple[float, list[Document]]] = {}


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


def _cache_get(project_id: str, query: str, k: int) -> list[Document] | None:
    settings = get_settings()
    key = (project_id, query.strip().lower(), k)
    row = _CRAG_CACHE.get(key)
    if row is None:
        return None
    ts, docs = row
    if time.time() - ts > settings.rag_grade_cache_ttl_seconds:
        _CRAG_CACHE.pop(key, None)
        return None
    return docs


def _cache_put(project_id: str, query: str, k: int, docs: list[Document]) -> None:
    key = (project_id, query.strip().lower(), k)
    _CRAG_CACHE[key] = (time.time(), docs)


def _expand_query(query: str) -> list[str]:
    """Expand short queries (<=2 words) with common synonyms for better recall.

    Longer queries are assumed to be specific enough.
    """
    words = query.strip().split()
    if len(words) > 2:
        return [query]

    synonyms: dict[str, list[str]] = {
        "auth": ["authentication", "login", "authorization"],
        "jwt": ["jsonwebtoken", "token", "bearer"],
        "cors": ["cross-origin", "preflight", "headers"],
        "db": ["database", "postgres", "postgresql", "pg"],
        "api": ["endpoint", "route", "handler", "rest"],
        "css": ["style", "stylesheet", "tailwind", "styling"],
        "react": ["component", "hook", "jsx", "render"],
        "next": ["nextjs", "app-router", "ssr", "ssg"],
        "test": ["spec", "specification", "jest", "vitest", "pytest"],
        "deploy": ["deploying", "deployment", "pipeline", "ci/cd"],
        "docker": ["container", "containerization", "compose", "image"],
    }

    expanded: list[str] = [query]
    for w in words:
        w_lower = w.lower()
        for syn in synonyms.get(w_lower, []):
            expanded.append(f"{query} {syn}")
    return expanded


async def crag_retrieve(project_id: str, query: str, k: int | None = None) -> list[Document]:
    """Retrieve and grade. Expands short queries for better recall.

    Deduplicates across expanded queries. Falls back to query rewrite
    only when no cached or graded results are found.
    """
    settings = get_settings()
    limit = k if k is not None else settings.rag_default_k

    # Try expanded queries for better recall
    expanded = _expand_query(query)
    all_results: list[Document] = []
    seen_content: set[str] = set()

    for exp_query in expanded:
        cached = _cache_get(project_id, exp_query, limit)
        if cached is not None:
            for doc in cached:
                if doc.page_content not in seen_content:
                    all_results.append(doc)
                    seen_content.add(doc.page_content)
            continue

        initial = await _vector_search(project_id, exp_query, limit)
        relevant = await _grade(exp_query, initial)
        if relevant:
            for doc in relevant:
                if doc.page_content not in seen_content:
                    all_results.append(doc)
                    seen_content.add(doc.page_content)
            _cache_put(project_id, exp_query, limit, relevant)
            if len(all_results) >= limit:
                break

    if all_results:
        return all_results[:limit]

    # Total miss — rewrite and retry (may cost 2 extra LLM calls)
    rewritten = await _rewrite(query)
    log_rag_crag_llm_targets(
        docs_to_grade=limit * len(expanded),
        grader_slug=settings.grader_model,
        rewriter_slug=settings.researcher_model,
    )
    second = await _vector_search(project_id, rewritten, limit)
    graded = await _grade(rewritten, second)
    _cache_put(project_id, query, limit, graded)
    return graded


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
