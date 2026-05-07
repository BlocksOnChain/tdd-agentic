"""Project-scoped RAG ingestion: split, embed, and persist into Qdrant.

Uses ``AsyncQdrantClient`` directly so the whole pipeline is async and we
have full control over point IDs and payload shape. Point IDs are stable
UUID5s derived from ``(source, chunk)`` so re-ingesting the same content
just upserts in place.
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from backend.config import get_settings
from backend.rag.embeddings import embedding_dim, get_embeddings


# Stable namespace so two processes generate identical UUIDs for identical content.
_RAG_NAMESPACE = uuid.UUID("d4f8a1c2-1234-5678-9abc-def012345678")

# Payload key used for chunk text — must match what retrieval reads back.
PAYLOAD_TEXT_KEY = "page_content"


def collection_name(project_id: str) -> str:
    return f"project_{project_id.replace('-', '_')}"


def _stable_id(text: str, source: str) -> str:
    return str(uuid.uuid5(_RAG_NAMESPACE, f"{source}::{text}"))


async def ensure_collection(project_id: str) -> str:
    """Create the Qdrant collection for a project if it doesn't exist."""
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        name = collection_name(project_id)
        existing = {c.name for c in (await client.get_collections()).collections}
        if name not in existing:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=embedding_dim(), distance=Distance.COSINE),
            )
        return name
    finally:
        await client.close()


async def ingest_text(
    project_id: str,
    text: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    """Chunk, embed, and upsert ``text`` into the project's collection.

    Returns the number of chunks written.
    """
    name = await ensure_collection(project_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    if not chunks:
        return 0

    base_meta = {"source": source, "project_id": project_id, **(metadata or {})}
    embeddings = get_embeddings()
    vectors = await embeddings.aembed_documents(chunks)

    points = [
        PointStruct(
            id=_stable_id(chunk, source),
            vector=vec,
            payload={PAYLOAD_TEXT_KEY: chunk, **base_meta},
        )
        for chunk, vec in zip(chunks, vectors)
    ]

    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        await client.upsert(collection_name=name, points=points)
        return len(points)
    finally:
        await client.close()


async def ingest_document_file(project_id: str, path: str) -> int:
    """Read a local file and ingest its contents."""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    return await ingest_text(project_id, text, source=str(path))
