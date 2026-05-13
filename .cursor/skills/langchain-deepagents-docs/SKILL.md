---
name: langchain-deepagents-docs
description: Consult and summarize official LangChain DeepAgents documentation and LangChain MCP documentation. Use when the user mentions LangChain DeepAgents/deepagents, LangChain MCP, deep agents, agent graphs/subgraphs, or asks how to implement or integrate DeepAgents or MCP based on LangChain docs.
disable-model-invocation: true
---

# LangChain DeepAgents docs

## Quick start

When asked about LangChain DeepAgents or LangChain MCP, use the official docs as the source of truth:

- LangChain MCP docs entry point: `https://docs.langchain.com/mcp`
- LangChain DeepAgents Python docs entry point: `https://docs.langchain.com/oss/python/deepagents`

## Workflow

1. **Pin the user’s intent**
   - Are they asking for a conceptual explanation, API usage, implementation guidance, or debugging?
2. **Fetch the relevant doc pages**
   - Start from the two entry points above.
   - Follow links within the docs to the exact feature being discussed (e.g., agent composition, subgraphs, tools, adapters, MCP integration).
3. **Summarize with fidelity**
   - Prefer quoting or tightly paraphrasing what the docs say.
   - If the docs don’t cover a detail, say so and propose a safe default or a next verification step.
4. **Translate docs → actionable output**
   - If the user wants code changes, implement the smallest correct change that matches the docs and the project’s patterns.
   - If the user wants an architecture decision, list the doc-backed constraints and recommend an option.
5. **Record references**
   - Include the exact doc page URLs used (not just the entry points).
   - Mention the retrieval date in prose when accuracy matters (docs can change).

## Output expectations

- Use doc-backed terminology (don’t invent class/function names).
- Include links (wrapped in backticks or markdown links) to the specific pages you relied on.
- Avoid guesses: when uncertain, fetch more doc context from the official sources above.

## Additional resources

- See [reference.md](reference.md) for the canonical doc URLs and trigger terms.
