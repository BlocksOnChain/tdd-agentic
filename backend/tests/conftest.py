"""Test isolation.

``get_settings()`` is ``lru_cache``d over a ``Settings`` that reads the
developer's real ``.env``. Without isolation the suite's result depends on
whoever ran it — a locally-set ``OPENROUTER_API_KEY`` silently changed how model
slugs resolve, and ``WORKSPACE_ROOT`` defaulted to the container path ``/app``,
which does not exist on a dev machine.

Environment variables take precedence over ``.env`` in pydantic-settings, so
pinning a baseline here makes every run deterministic. Individual tests may
still override via ``monkeypatch.setenv`` plus ``get_settings.cache_clear()``.
"""
from __future__ import annotations

import os

import pytest

# Deterministic baseline for anything a test might read out of Settings.
_TEST_ENV: dict[str, str] = {
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "OPENROUTER_API_KEY": "",
    "OPENAI_BASE_URL": "",
    "TAVILY_API_KEY": "",
    "PM_MODEL": "anthropic/claude-sonnet-4-6",
    "RESEARCHER_MODEL": "openai/gpt-4o",
    "LEAD_MODEL": "anthropic/claude-sonnet-4-6",
    "DEV_MODEL": "anthropic/claude-sonnet-4-6",
    "GRADER_MODEL": "anthropic/claude-haiku-4-5",
    "BACKEND_DEV_MODEL": "",
    "FRONTEND_DEV_MODEL": "",
    "COORDINATOR_MODEL": "",
    "DEVOPS_MODEL": "",
    "QA_MODEL": "",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
}


@pytest.fixture(scope="session", autouse=True)
def hermetic_env(tmp_path_factory):
    """Pin the environment for the whole session, restoring it afterwards."""
    workspace = tmp_path_factory.mktemp("workspace")
    env = {**_TEST_ENV, "WORKSPACE_ROOT": str(workspace)}

    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)

    from backend.config import get_settings

    get_settings.cache_clear()
    yield
    for key, prior in saved.items():
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def no_database_writes(monkeypatch):
    """Stub agent-log persistence so no test opens a Postgres connection.

    ``emit`` publishes to the EventBus, which queues a row for the background
    log writer. Without this, any test that drives a real agent node tries to
    connect to localhost:5432 and floods the output with connection errors.
    Tests that care about persistence patch this themselves.
    """
    import backend.agent_logs.persist as persist_mod

    async def _noop(events):
        return None

    monkeypatch.setattr(persist_mod, "persist_agent_events", _noop)
    monkeypatch.setattr(persist_mod, "persist_agent_event", lambda event: _noop([event]))
    yield


@pytest.fixture(autouse=True)
def reset_caches():
    """Drop process-wide caches so tests can't leak state into one another."""
    from backend.agents.graph import clear_graph_cache
    from backend.agents.llm import reset_chat_model_cache
    from backend.agents.skills import loader
    from backend.rag.retrieval import clear_crag_cache

    def _clear() -> None:
        reset_chat_model_cache()
        clear_graph_cache()
        clear_crag_cache()
        loader._inject_cache.clear()

    _clear()
    yield
    _clear()
