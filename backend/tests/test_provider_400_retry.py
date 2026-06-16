from __future__ import annotations

import asyncio

import pytest


class Provider400Error(Exception):
    status_code = 400


class DummyRunnable:
    """Minimal Runnable-like object for retry wrapper tests."""

    def __init__(self, *, fail_times: int, message: str):
        self._fail_times = fail_times
        self._calls = 0
        self._message = message

    def invoke(self, _input, config=None, **_kwargs):  # noqa: ANN001
        self._calls += 1
        if self._calls <= self._fail_times:
            raise Provider400Error(self._message)
        return "ok"

    async def ainvoke(self, _input, config=None, **_kwargs):  # noqa: ANN001
        return self.invoke(_input, config=config, **_kwargs)


def test_provider_400_retries_once_after_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.agents import llm as llm_mod

    slept: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(llm_mod, "_sleep_sync", _fake_sleep)
    monkeypatch.setattr(llm_mod, "_sleep_async", lambda seconds: asyncio.sleep(0))

    runnable = DummyRunnable(fail_times=1, message="Provider returned error: upstream timeout")
    wrapped = llm_mod.with_retry(runnable)

    assert wrapped.invoke("x") == "ok"
    assert slept == [60.0]


def test_provider_400_crashes_on_second_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.agents import llm as llm_mod

    slept: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(llm_mod, "_sleep_sync", _fake_sleep)

    runnable = DummyRunnable(fail_times=2, message="provider error: overloaded")
    wrapped = llm_mod.with_retry(runnable)

    with pytest.raises(Provider400Error):
        wrapped.invoke("x")

    # Only one retry is allowed.
    assert slept == [60.0]


def test_generic_400_does_not_sleep_or_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.agents import llm as llm_mod

    slept: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(llm_mod, "_sleep_sync", _fake_sleep)

    runnable = DummyRunnable(fail_times=1, message="Bad Request: invalid payload")
    wrapped = llm_mod.with_retry(runnable)

    with pytest.raises(Provider400Error):
        wrapped.invoke("x")

    assert slept == []


class Transient503Error(Exception):
    status_code = 503


def test_transient_503_retries_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.agents import llm as llm_mod

    slept: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(llm_mod, "_sleep_sync", _fake_sleep)
    monkeypatch.setattr(llm_mod, "_backoff_seconds", lambda _attempt: 0.01)

    class FlakyRunnable:
        def __init__(self) -> None:
            self._calls = 0

        def invoke(self, _input, config=None, **_kwargs):  # noqa: ANN001
            self._calls += 1
            if self._calls == 1:
                raise Transient503Error("upstream overloaded")
            return "ok"

        async def ainvoke(self, _input, config=None, **_kwargs):  # noqa: ANN001
            return self.invoke(_input, config=config, **_kwargs)

    wrapped = llm_mod.with_retry(FlakyRunnable())
    assert wrapped.invoke("x") == "ok"
    assert len(slept) == 1

