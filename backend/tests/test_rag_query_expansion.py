"""Tests for RAG query expansion logic."""
from __future__ import annotations

from backend.rag.retrieval import _expand_query


def test_short_query_expands() -> None:
    """Single-word queries are expanded with synonyms."""
    result = _expand_query("auth")
    assert "auth" in result
    assert "auth authentication" in result
    assert "auth login" in result


def test_two_word_query_expands() -> None:
    """Two-word queries are expanded with synonyms for each word."""
    result = _expand_query("jwt auth")
    assert "jwt auth" in result
    # Should have expansions for both words
    assert any("jwt jsonwebtoken" in r for r in result)
    assert any("jwt token" in r for r in result)


def test_long_query_not_expanded() -> None:
    """Queries with more than 2 words are returned as-is."""
    query = "implement jwt auth middleware"
    result = _expand_query(query)
    assert len(result) == 1
    assert result[0] == query


def test_empty_query() -> None:
    """Empty or whitespace queries handled gracefully."""
    assert _expand_query("") == [""]
    assert _expand_query("  ") == [""]


def test_no_synonyms_for_unknown_word() -> None:
    """Unknown words produce only the original query."""
    result = _expand_query("xyz")
    assert len(result) == 1
    assert result[0] == "xyz"


def test_multiple_synonyms_added() -> None:
    """Each synonym adds an expanded query variant."""
    result = _expand_query("react")
    # Original + each synonym
    assert len(result) >= 2  # At minimum: "react" + "react component"
    assert "react" in result


def test_css_expansion() -> None:
    """CSS query expands to styling-related terms."""
    result = _expand_query("css")
    assert any("style" in r for r in result)
    assert any("tailwind" in r for r in result)
