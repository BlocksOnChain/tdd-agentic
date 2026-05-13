"""Per-role OpenAI base URL resolution (researcher + grader overrides)."""
from __future__ import annotations

from types import SimpleNamespace

from backend.agents import llm


def test_researcher_platform_openai_ignores_global_base() -> None:
    s = SimpleNamespace(
        researcher_openai_base_url="",
        researcher_use_platform_openai=True,
        grader_openai_base_url="",
        grader_use_platform_openai=False,
        openai_base_url="http://127.0.0.1:8080/v1",
    )
    assert llm._openai_base_url_for_client(s, host_override="researcher") is None


def test_researcher_explicit_base_url_wins_over_platform_flag() -> None:
    s = SimpleNamespace(
        researcher_openai_base_url="https://api.openai.com/v1",
        researcher_use_platform_openai=False,
        grader_openai_base_url="",
        grader_use_platform_openai=False,
        openai_base_url="http://127.0.0.1:8080/v1",
    )
    assert llm._openai_base_url_for_client(s, host_override="researcher") == "https://api.openai.com/v1"


def test_researcher_inherits_global_when_no_override() -> None:
    s = SimpleNamespace(
        researcher_openai_base_url="",
        researcher_use_platform_openai=False,
        grader_openai_base_url="",
        grader_use_platform_openai=False,
        openai_base_url="http://127.0.0.1:8080/v1",
    )
    assert llm._openai_base_url_for_client(s, host_override="researcher") == "http://127.0.0.1:8080/v1"


def test_none_path_ignores_role_overrides() -> None:
    s = SimpleNamespace(
        researcher_openai_base_url="https://api.openai.com/v1",
        researcher_use_platform_openai=True,
        grader_openai_base_url="https://api.openai.com/v1",
        grader_use_platform_openai=True,
        openai_base_url="http://local/v1",
    )
    assert llm._openai_base_url_for_client(s, host_override="none") == "http://local/v1"


def test_grader_platform_openai_ignores_global_base() -> None:
    s = SimpleNamespace(
        researcher_openai_base_url="",
        researcher_use_platform_openai=False,
        grader_openai_base_url="",
        grader_use_platform_openai=True,
        openai_base_url="http://127.0.0.1:8080/v1",
    )
    assert llm._openai_base_url_for_client(s, host_override="grader") is None


def test_grader_explicit_base_url() -> None:
    s = SimpleNamespace(
        researcher_openai_base_url="",
        researcher_use_platform_openai=False,
        grader_openai_base_url="https://api.openai.com/v1",
        grader_use_platform_openai=False,
        openai_base_url="http://127.0.0.1:8080/v1",
    )
    assert llm._openai_base_url_for_client(s, host_override="grader") == "https://api.openai.com/v1"
