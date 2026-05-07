"""Web search tool powered by Claude's server-side ``web_search`` tool.

Exposes a regular LangChain ``@tool`` so any agent (including a non-Claude
researcher) can call ``web_search(query)``. Internally we invoke a Claude
model that has Anthropic's server-side ``web_search_20250305`` tool bound,
so search execution happens on Anthropic's infrastructure with no extra
API key required.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from backend.config import get_settings


def _extract_text(content: Any) -> str:
    """Pull text out of a Claude response that mixes text and tool blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "\n\n".join(p for p in parts if p)
    return str(content)


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for recent or authoritative information.

    Returns a concise textual report of the top findings with inline source
    URLs. Powered by Claude's server-side web search — no separate search
    API key required.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        return json.dumps({"error": "ANTHROPIC_API_KEY not configured"})

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return json.dumps({"error": "langchain-anthropic not installed"})

    try:
        llm = ChatAnthropic(
            model=settings.web_search_model,
            api_key=settings.anthropic_api_key,
            max_tokens=2048,
        )
        llm_with_search = llm.bind_tools(
            [
                {
                    "type": settings.web_search_tool_version,
                    "name": "web_search",
                    "max_uses": max(1, min(int(max_results), 10)),
                    # Haiku models require an explicit allowed_callers list
                    # because they don't support programmatic tool calling.
                    "allowed_callers": ["direct"],
                }
            ]
        )
        response = await llm_with_search.ainvoke(
            "Search the web for the following query and report the most relevant, "
            "specific findings with concrete details and source URLs. "
            "Prefer authoritative sources (official docs, primary sources, well-known publications).\n\n"
            f"Query: {query}"
        )
        text = _extract_text(response.content)
        return text or json.dumps({"error": "no results returned"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
