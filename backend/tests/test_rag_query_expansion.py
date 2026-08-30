"""RAG query expansion.

Expansion substitutes a synonym for the matched word rather than appending it —
"jwt auth" becomes "jsonwebtoken auth", not "jwt auth jsonwebtoken". Appending
drifted the embedding away from both the original term and the synonym, which is
the opposite of what expansion is for. The fan-out is capped because each
variant costs a vector search plus a grader call.
"""
from __future__ import annotations

from backend.rag.retrieval import MAX_QUERY_EXPANSIONS, _expand_query


def test_original_query_is_always_first() -> None:
    for q in ("auth", "jwt auth", "xyz"):
        assert _expand_query(q)[0] == q


def test_short_query_expands_by_substitution() -> None:
    result = _expand_query("auth")
    assert "auth" in result
    assert "authentication" in result
    # The old append form must not come back.
    assert "auth authentication" not in result


def test_two_word_query_substitutes_the_matched_word() -> None:
    result = _expand_query("jwt auth")
    assert result[0] == "jwt auth"
    assert "jsonwebtoken auth" in result
    # The untouched word is preserved in every variant.
    assert all(len(r.split()) == 2 for r in result)


def test_expansion_is_capped() -> None:
    """Every extra variant is another search + grader call — bound the fan-out."""
    # "auth" alone has 3 synonyms and "db" has 4; both would overflow the cap.
    for q in ("auth", "db", "test deploy"):
        assert len(_expand_query(q)) <= MAX_QUERY_EXPANSIONS


def test_expansions_are_unique() -> None:
    result = _expand_query("test test")
    assert len(result) == len(set(result))


def test_long_query_not_expanded() -> None:
    query = "implement jwt auth middleware"
    assert _expand_query(query) == [query]


def test_empty_query() -> None:
    assert _expand_query("") == [""]
    assert _expand_query("  ") == [""]


def test_no_synonyms_for_unknown_word() -> None:
    assert _expand_query("xyz") == ["xyz"]
