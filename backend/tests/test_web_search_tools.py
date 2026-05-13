"""Researcher web_search backend selection and Tavily calls."""
from __future__ import annotations

import json

import pytest

from backend.tools import web_search_tools as ws


@pytest.mark.asyncio
async def test_web_search_uses_tavily_when_configured(monkeypatch) -> None:
    class _Settings:
        web_search_provider = "tavily"
        tavily_api_key = "tvly-test"
        anthropic_api_key = ""

    async def fake_tavily(query: str, max_results: int, api_key: str) -> str:
        assert api_key == "tvly-test"
        assert query == "shadcn todo animations"
        return "Summary: found sources"

    monkeypatch.setattr(ws, "get_settings", lambda: _Settings())
    monkeypatch.setattr(ws, "_web_search_tavily", fake_tavily)

    result = await ws.web_search.ainvoke({"query": "shadcn todo animations"})
    assert "found sources" in result


def test_resolve_backend_prefers_tavily_when_key_present(monkeypatch) -> None:
    class _Settings:
        web_search_provider = "auto"
        tavily_api_key = "tvly-test"

    monkeypatch.setattr(ws, "get_settings", lambda: _Settings())
    assert ws._resolve_backend() == "tavily"
