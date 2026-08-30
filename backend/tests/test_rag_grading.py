"""CRAG document grading.

Regression: ``_grade`` awaited one grader call per document, inside a loop that
itself ran once per expanded query — up to 16 strictly sequential, rate-limited
calls for a single ``rag_query``. Grading is now one batched call with a
concurrent per-document fallback.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from backend.rag import retrieval


def _docs(n: int) -> list[Document]:
    return [Document(page_content=f"doc-{i}", metadata={"source": f"s{i}"}) for i in range(n)]


class _Batch(SimpleNamespace):
    """Stands in for _BatchRelevanceScore (has .relevant)."""


class _Single(SimpleNamespace):
    """Stands in for _RelevanceScore (has .is_relevant)."""


class _Grader:
    """Stub structured-output grader that records how many calls it received.

    ``responses`` is a callable taking the 1-based call index. Deliberately not a
    MagicMock: MagicMock is callable and iterable, which silently turns a wrong
    stub into a plausible-looking empty result.
    """

    def __init__(self, responses, *, fail: bool = False):
        self.responses = responses
        self.fail = fail
        self.calls = 0

    async def ainvoke(self, messages, *a, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("grader unavailable")
        return self.responses(self.calls)


def _patch_grader(grader):
    model = MagicMock()
    model.with_structured_output.return_value = grader
    return patch.object(retrieval, "grader_model", return_value=model)


@pytest.mark.asyncio
async def test_batch_grading_uses_one_call_for_many_docs() -> None:
    grader = _Grader(lambda _n: _Batch(relevant=[True, False, True, False]))
    with _patch_grader(grader):
        kept = await retrieval._grade("jwt auth", _docs(4))

    assert grader.calls == 1, "grading 4 docs must not cost 4 model calls"
    assert [d.page_content for d in kept] == ["doc-0", "doc-2"]


@pytest.mark.asyncio
async def test_empty_input_costs_no_call() -> None:
    grader = _Grader(lambda _n: _Batch(relevant=[]))
    with _patch_grader(grader):
        assert await retrieval._grade("q", []) == []
    assert grader.calls == 0


@pytest.mark.asyncio
async def test_wrong_length_verdict_falls_back_to_per_document() -> None:
    """A small grader that returns the wrong number of booleans must not corrupt results."""
    # First (batch) call returns 2 verdicts for 3 docs; the fallback path then
    # grades each document individually.
    def _result(call_index: int):
        if call_index == 1:
            return _Batch(relevant=[True, False])
        return _Single(is_relevant=True)

    grader = _Grader(_result)
    with _patch_grader(grader):
        kept = await retrieval._grade("q", _docs(3))

    assert grader.calls == 4  # 1 batch attempt + 3 individual
    assert len(kept) == 3


@pytest.mark.asyncio
async def test_grader_failure_keeps_documents() -> None:
    """A grader outage must not silently empty the context block."""
    grader = _Grader(lambda _n: None, fail=True)
    with _patch_grader(grader):
        kept = await retrieval._grade("q", _docs(3))

    assert [d.page_content for d in kept] == ["doc-0", "doc-1", "doc-2"]


@pytest.mark.asyncio
async def test_individual_grading_runs_concurrently() -> None:
    """The fallback path must fan out, not serialize."""
    started = 0
    peak = 0

    class _SlowGrader:
        async def ainvoke(self, messages, *a, **kw):
            nonlocal started, peak
            started += 1
            peak = max(peak, started)
            await asyncio.sleep(0.02)
            started -= 1
            return _Single(is_relevant=True)

    with _patch_grader(_SlowGrader()):
        kept = await retrieval._grade_individually("q", _docs(5))

    assert len(kept) == 5
    assert peak > 1, "per-document grading should overlap, not run one at a time"


def test_crag_cache_is_bounded() -> None:
    retrieval.clear_crag_cache()
    for i in range(retrieval.CRAG_CACHE_MAX_ENTRIES + 50):
        retrieval._cache_put("proj", f"query-{i}", 4, _docs(1))

    assert len(retrieval._CRAG_CACHE) == retrieval.CRAG_CACHE_MAX_ENTRIES
    # Least-recently-used entries were the ones dropped.
    assert retrieval._cache_get("proj", "query-0", 4) is None
    assert retrieval._cache_get("proj", f"query-{retrieval.CRAG_CACHE_MAX_ENTRIES + 49}", 4) is not None
    retrieval.clear_crag_cache()


def test_crag_cache_refreshes_recency_on_read() -> None:
    retrieval.clear_crag_cache()
    for i in range(retrieval.CRAG_CACHE_MAX_ENTRIES):
        retrieval._cache_put("proj", f"q{i}", 4, _docs(1))

    # Touch the oldest entry, then overflow by one.
    assert retrieval._cache_get("proj", "q0", 4) is not None
    retrieval._cache_put("proj", "new", 4, _docs(1))

    assert retrieval._cache_get("proj", "q0", 4) is not None, "recently read entry was evicted"
    assert retrieval._cache_get("proj", "q1", 4) is None
    retrieval.clear_crag_cache()


def test_grader_prompts_declare_only_their_real_variables() -> None:
    """Braces in literal JSON examples must be escaped.

    Regression: ``{is_relevant: true/false}`` in the system prompt registered a
    phantom template variable, so every format_messages call raised KeyError —
    swallowed by the grader's except, which kept the document. Relevance
    filtering was silently a no-op.
    """
    assert set(retrieval._GRADER_PROMPT.input_variables) == {"query", "document"}
    assert set(retrieval._BATCH_GRADER_PROMPT.input_variables) == {"query", "documents", "n"}


def test_grader_prompts_actually_format() -> None:
    single = retrieval._GRADER_PROMPT.format_messages(query="q", document="d")
    assert any("is_relevant" in str(m.content) for m in single)

    batch = retrieval._BATCH_GRADER_PROMPT.format_messages(query="q", documents="[1] d", n=1)
    assert any("relevant" in str(m.content) for m in batch)
