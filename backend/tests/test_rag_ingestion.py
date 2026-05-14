"""RAG / Qdrant ingestion edge cases."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.rag.ingestion import _is_collection_already_exists_error, collection_name, ensure_collection


def test_is_collection_already_exists_error() -> None:
    h = httpx.Headers({})
    assert _is_collection_already_exists_error(UnexpectedResponse(409, "Conflict", b"", h)) is True
    assert _is_collection_already_exists_error(
        UnexpectedResponse(400, "Bad Request", b'{"status":{"error":"already exists!"}}', h)
    ) is True
    assert _is_collection_already_exists_error(UnexpectedResponse(400, "Bad Request", b"no", h)) is False
    assert _is_collection_already_exists_error(UnexpectedResponse(500, "Error", b"", h)) is False


@pytest.mark.asyncio
async def test_ensure_collection_swallows_concurrent_create_conflict() -> None:
    """If create_collection races and Qdrant returns 409, treat as success."""
    pid = "606ec804-eec1-4801-bafe-6e0d6f88f159"
    name = collection_name(pid)
    h = httpx.Headers({})

    mock_client = MagicMock()
    mock_client.get_collections = AsyncMock(
        return_value=MagicMock(collections=[MagicMock(name="other")])
    )
    mock_client.create_collection = AsyncMock(
        side_effect=UnexpectedResponse(
            409,
            "Conflict",
            b'{"status":{"error":"Wrong input: Collection already exists!"}}',
            h,
        )
    )
    mock_client.close = AsyncMock()

    with patch("backend.rag.ingestion.AsyncQdrantClient", return_value=mock_client):
        assert await ensure_collection(pid) == name

    mock_client.create_collection.assert_awaited_once()
