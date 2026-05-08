"""Web search tool for the researcher.

Two backends:

1. **Anthropic** — Claude with server-side ``web_search`` (``WEB_SEARCH_MODEL`` /
   ``WEB_SEARCH_TOOL_VERSION`` matter only here). Uses ``ANTHROPIC_API_KEY``.

2. **Tavily** — direct HTTP calls to Tavily's search API. Works alongside a
**local chat model** because search is unrelated to Llama/tool versions.

``WEB_SEARCH_PROVIDER`` (``auto`` | ``anthropic`` | ``tavily``) selects the
backend; ``auto`` prefers Tavily when ``TAVILY_API_KEY`` is set.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
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


def _format_tavily_response(data: dict[str, Any], query: str) -> str:
    lines: list[str] = []
    ans = data.get("answer")
    if isinstance(ans, str) and ans.strip():
        lines.append(f"Summary: {ans.strip()}")

    results = data.get("results") or []
    if not isinstance(results, list):
        results = []

    lines.append("")
    lines.append(f"Top sources for query: {query!r}")
    for i, item in enumerate(results[:10], start=1):
        if not isinstance(item, dict):
            continue
        title = item.get("title") or "(no title)"
        url = item.get("url") or ""
        body = item.get("content") or ""
        lines.append("")
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if isinstance(body, str) and body.strip():
            excerpt = body.strip().replace("\n", " ")
            if len(excerpt) > 500:
                excerpt = excerpt[:500] + "…"
            lines.append(f"   {excerpt}")

    text = "\n".join(lines).strip()
    return text or json.dumps({"error": "empty Tavily payload", "query": query})


async def _web_search_tavily(query: str, max_results: int, api_key: str) -> str:
    n = max(1, min(int(max_results), 10))
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": n,
        "include_answer": True,
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post("https://api.tavily.com/search", json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": f"Tavily HTTP {exc.response.status_code}", "detail": exc.response.text[:500]}
            )
        except Exception as exc:  # pragma: no cover
            return json.dumps({"error": str(exc)})
    try:
        data = resp.json()
    except Exception as exc:
        return json.dumps({"error": f"invalid Tavily JSON: {exc}"})
    return _format_tavily_response(data, query)


async def _web_search_anthropic(query: str, max_results: int) -> str:
    settings = get_settings()
    if not settings.anthropic_api_key:
        return json.dumps({"error": "ANTHROPIC_API_KEY not configured"})

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return json.dumps({"error": "langchain-anthropic not installed"})

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


def _resolve_backend() -> str:
    settings = get_settings()
    raw = (settings.web_search_provider or "auto").strip().lower()
    if raw not in {"auto", "anthropic", "tavily"}:
        return "auto"
    if raw != "auto":
        return raw
    if settings.tavily_api_key.strip():
        return "tavily"
    return "anthropic"


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for recent or authoritative information.

    Returns findings with URLs. Backend is Tavily (``TAVILY_API_KEY``) or
    Claude server-side search (``ANTHROPIC_API_KEY``), see ``WEB_SEARCH_PROVIDER``.
    """
    settings = get_settings()
    backend = _resolve_backend()

    try:
        if backend == "tavily":
            key = settings.tavily_api_key.strip()
            if not key:
                return json.dumps(
                    {
                        "error": "WEB_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is empty",
                    }
                )
            return await _web_search_tavily(query, max_results, key)

        if not settings.anthropic_api_key:
            return json.dumps(
                {
                    "error": "Anthropic web search needs ANTHROPIC_API_KEY; for local LLMs set TAVILY_API_KEY and WEB_SEARCH_PROVIDER=tavily or auto",
                }
            )
        return await _web_search_anthropic(query, max_results)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
