"""Embedding model factory — OpenAI by default, local sentence-transformers as fallback."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from backend.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key or None,
        )

    if provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)

    raise ValueError(f"Unsupported embedding_provider '{provider}'")


@lru_cache(maxsize=1)
def embedding_dim() -> int:
    """Return the embedding vector size for the active provider."""
    settings = get_settings()
    if settings.embedding_provider.lower() == "openai":
        # text-embedding-3-small=1536, text-embedding-3-large=3072
        return 3072 if "large" in settings.embedding_model else 1536
    # MiniLM-L6-v2 = 384, MPNet-base-v2 = 768
    return 384 if "MiniLM" in settings.local_embedding_model else 768
