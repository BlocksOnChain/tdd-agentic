"""CRAG-style retrieval pipeline: retrieve → grade → (rewrite | return).

Implemented directly against ``AsyncQdrantClient`` for full async control.
A small grader LLM filters out irrelevant chunks. If nothing relevant
survives, the query is rewritten and we retry once before giving up.

Query expansion: for single-word queries, auto-expand with related terms
for better recall.

TTL-aware cache with LRU eviction.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
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


class _BatchRelevanceScore(BaseModel):
    """Relevance verdicts for a numbered batch of documents, in order."""

    relevant: list[bool] = Field(
        description="One true/false per document, in the order the documents were given"
    )


# NOTE: braces in the literal JSON example MUST be doubled. ChatPromptTemplate
# parses single braces as variable slots — an unescaped ``{is_relevant: ...}``
# registered a phantom input variable and made every ``format_messages`` call
# raise KeyError, which the grader's broad ``except`` then swallowed by keeping
# the document. The relevance filter silently passed everything through.
_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether a retrieved document is relevant to a user query. "
            'Reply with a JSON object {{"is_relevant": true}} or {{"is_relevant": false}}. '
            "Be strict — only mark relevant if the document contains information that "
            "helps answer the query.",
        ),
        ("human", "Query:\n{query}\n\nDocument:\n{document}"),
    ]
)

_BATCH_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade which retrieved documents are relevant to a user query. "
            "You are given {n} numbered documents. Reply with a JSON object "
            '{{"relevant": [...]}} containing EXACTLY {n} booleans, one per '
            "document, in the same order. Be strict — mark a document relevant "
            "only if it contains information that helps answer the query.",
        ),
        ("human", "Query:\n{query}\n\nDocuments:\n{documents}"),
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

_logger = logging.getLogger(__name__)

# Characters of each document shown to the grader.
GRADER_DOC_CHARS = 2000

# Bounded TTL cache. Entries expire by time AND the map is capped by size — the
# previous dict only dropped an expired entry when that exact key was requested
# again, so a long run accumulated every (project, query, k) it had ever seen
# and held the Document lists for the process lifetime.
CRAG_CACHE_MAX_ENTRIES = 256
_CRAG_CACHE: OrderedDict[tuple[str, str, int], tuple[float, list[Document]]] = OrderedDict()

# One Qdrant client for the process. Constructing one per search opened a fresh
# HTTP connection for every vector lookup.
_qdrant_client: AsyncQdrantClient | None = None
_qdrant_lock = asyncio.Lock()


async def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        async with _qdrant_lock:
            if _qdrant_client is None:
                _qdrant_client = AsyncQdrantClient(url=get_settings().qdrant_url)
    return _qdrant_client


async def close_qdrant_client() -> None:
    global _qdrant_client
    if _qdrant_client is not None:
        client, _qdrant_client = _qdrant_client, None
        await client.close()


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

    client = await get_qdrant_client()
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


async def _grade(query: str, docs: list[Document]) -> list[Document]:
    """Filter ``docs`` down to those relevant to ``query`` — in ONE model call.

    This used to await one grader call per document, inside a loop that itself
    ran once per expanded query. With ``k=4`` and a two-word query that is up to
    16 strictly sequential calls for a single ``rag_query``, all sharing the
    agent's own provider rate-limit bucket — a floor of tens of seconds before
    any inference time. Grading the whole batch in one call collapses that to a
    single round trip, and a per-document fallback keeps the behaviour correct
    if a small local grader can't hold the batch format.
    """
    if not docs:
        return []
    if len(docs) == 1:
        return await _grade_individually(query, docs)

    grader = grader_model().with_structured_output(_BatchRelevanceScore)
    numbered = "\n\n".join(
        f"[{i + 1}]\n{doc.page_content[:GRADER_DOC_CHARS]}" for i, doc in enumerate(docs)
    )
    try:
        verdict = await grader.ainvoke(
            _BATCH_GRADER_PROMPT.format_messages(
                query=query, documents=numbered, n=len(docs)
            )
        )
        flags = list(verdict.relevant)
        if len(flags) == len(docs):
            return [doc for doc, keep in zip(docs, flags) if keep]
        _logger.warning(
            "batch grader returned %s verdicts for %s docs; grading individually",
            len(flags),
            len(docs),
        )
    except Exception:
        _logger.warning("batch grader failed; grading individually", exc_info=True)

    return await _grade_individually(query, docs)


async def _grade_individually(query: str, docs: list[Document]) -> list[Document]:
    """Per-document grading, run concurrently. Fallback path for _grade."""
    grader = grader_model().with_structured_output(_RelevanceScore)

    async def _one(doc: Document) -> Document | None:
        try:
            score = await grader.ainvoke(
                _GRADER_PROMPT.format_messages(
                    query=query, document=doc.page_content[:GRADER_DOC_CHARS]
                )
            )
            return doc if score.is_relevant else None
        except Exception:
            # A grader failure must not drop a document that may be relevant —
            # but it must be visible. A silent keep-everything here is
            # indistinguishable from a working filter, and hid a prompt-template
            # bug that disabled relevance grading entirely.
            _logger.warning("grader call failed; keeping document", exc_info=True)
            return doc

    results = await asyncio.gather(*(_one(d) for d in docs))
    return [d for d in results if d is not None]


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
    _CRAG_CACHE.move_to_end(key)  # LRU recency
    return docs


def _cache_put(project_id: str, query: str, k: int, docs: list[Document]) -> None:
    key = (project_id, query.strip().lower(), k)
    _CRAG_CACHE[key] = (time.time(), docs)
    _CRAG_CACHE.move_to_end(key)
    while len(_CRAG_CACHE) > CRAG_CACHE_MAX_ENTRIES:
        _CRAG_CACHE.popitem(last=False)  # evict least-recently-used


def clear_crag_cache() -> None:
    _CRAG_CACHE.clear()


MAX_QUERY_EXPANSIONS = 3


def _expand_query(query: str) -> list[str]:
    """Expand short queries (<=2 words) with common synonyms for better recall.

    Longer queries are assumed to be specific enough.

    Each variant *substitutes* the synonym for the matched word rather than
    appending it: for "jwt auth", "jsonwebtoken auth" is a sensible alternate
    embedding, while the old "jwt auth jsonwebtoken" drifted away from both. The
    original query always comes first, and the total is capped — every extra
    variant is another vector search plus another grader call.
    """
    stripped = query.strip()
    if not stripped:
        return [stripped]

    words = stripped.split()
    if len(words) > 2:
        return [stripped]

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

    expanded: list[str] = [stripped]
    seen = {stripped.lower()}
    for i, w in enumerate(words):
        for syn in synonyms.get(w.lower(), []):
            variant = " ".join([*words[:i], syn, *words[i + 1 :]])
            if variant.lower() in seen:
                continue
            seen.add(variant.lower())
            expanded.append(variant)
            if len(expanded) >= MAX_QUERY_EXPANSIONS:
                return expanded
    return expanded


async def crag_retrieve(project_id: str, query: str, k: int | None = None) -> list[Document]:
    """Retrieve and grade. Expands short queries for better recall.

    Deduplicates across expanded queries. Falls back to query rewrite
    only when no cached or graded results are found.
    """
    settings = get_settings()
    limit = k if k is not None else settings.rag_default_k

    # Try expanded queries for better recall. The variants are independent, so
    # they run concurrently — serially they multiplied the (already
    # rate-limited) search + grade round trip by the number of expansions.
    expanded = _expand_query(query)

    async def _retrieve_one(exp_query: str) -> list[Document]:
        cached = _cache_get(project_id, exp_query, limit)
        if cached is not None:
            return cached
        initial = await _vector_search(project_id, exp_query, limit)
        relevant = await _grade(exp_query, initial)
        if relevant:
            _cache_put(project_id, exp_query, limit, relevant)
        return relevant

    per_query = await asyncio.gather(
        *(_retrieve_one(q) for q in expanded), return_exceptions=True
    )

    # Merge in expansion order so the original query's hits rank first.
    all_results: list[Document] = []
    seen_content: set[str] = set()
    for exp_query, docs in zip(expanded, per_query):
        if isinstance(docs, BaseException):
            _logger.warning("expansion %r failed: %s", exp_query, docs)
            continue
        for doc in docs:
            if doc.page_content in seen_content:
                continue
            seen_content.add(doc.page_content)
            all_results.append(doc)

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
