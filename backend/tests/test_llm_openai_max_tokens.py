"""OpenAI-compatible chat models must cap completion length."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.agents.llm import get_chat_model


def test_openai_chat_model_receives_max_output_tokens() -> None:
    with patch("backend.agents.llm.get_settings") as settings_mock:
        settings = MagicMock()
        settings.max_output_tokens = 1024
        settings.openai_api_key = "test-key"
        settings.openai_request_timeout = 60.0
        settings.openai_base_url = "http://127.0.0.1:8080/v1"
        settings.openai_requests_per_second = 1.0
        settings.openai_burst = 2
        settings.llm_max_retries = 1
        settings_mock.return_value = settings

        with patch("backend.agents.llm.ChatOpenAI") as chat_ctor:
            get_chat_model("openai/test-model")
            kwargs = chat_ctor.call_args.kwargs
            assert kwargs["max_tokens"] == 1024
