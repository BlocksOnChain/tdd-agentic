"""In-process caches for RAG retrieval."""
from __future__ import annotations

_CRAG_CACHE: dict[tuple[str, str, int], tuple[float, list]] = {}


def crag_cache_get(key: tuple[str, str, int]) -> tuple[float, list] | None:
    return _CRAG_CACHE.get(key)


def crag_cache_put(key: tuple[str, str, int], value: tuple[float, list]) -> None:
    _CRAG_CACHE[key] = value


def crag_cache_pop(key: tuple[str, str, int]) -> None:
    _CRAG_CACHE.pop(key, None)


def invalidate_crag_cache_for_project(project_id: str) -> None:
    """Drop cached CRAG results after new vectors are ingested."""
    keys = [key for key in _CRAG_CACHE if key[0] == project_id]
    for key in keys:
        _CRAG_CACHE.pop(key, None)
