# LangChain Deep Agents (DeepAgents) – quick index for this repo

This is a short, repo-local index of the LangChain **Deep Agents** docs and the parts we rely on in `backend/agents/deep/`.

## Key upstream docs

- **Customization surface (`create_deep_agent`)**: `https://docs.langchain.com/oss/python/deepagents/customization`
  - Core knobs we use: `model`, `tools`, `system_prompt`, `backend`, `skills`, `memory`, `middleware`, `name`.
- **Subagents**: `https://docs.langchain.com/oss/python/deepagents/subagents`
  - Why: context quarantine; keep main context clean.
  - Note: Deep Agents auto-adds a synchronous `general-purpose` subagent unless replaced/disabled via harness profile.
- **Profiles (HarnessProfile)**: `https://docs.langchain.com/oss/python/deepagents/profiles`
  - Best-practice way to tune per-model behavior: prompt suffixes, excluded tools, excluded middleware, and general-purpose subagent settings.
- **Backends**: `https://docs.langchain.com/oss/python/deepagents/backends`
  - We use `FilesystemBackend(root_dir=..., virtual_mode=True)` to scope FS ops to `workspace_root/<project_id>`.

## How we apply this in tdd-agentic

- **Deep agent entrypoint**: `backend/agents/deep/adapter.py`
  - Builds one deep agent per specialist turn via `create_deep_agent(...)`.
  - Uses `FilesystemBackend` rooted to the per-project workspace.
  - Uses `memory=["/AGENTS.md"]` and `skills=["/_skills"]` when present.
  - Uses middleware for cross-cutting behavior (telemetry).

- **Tool telemetry (best-practice middleware)**: `backend/agents/deep/middleware.py`
  - Uses LangChain’s `wrap_tool_call` hook to emit `tool_call` and `tool_result` events.
  - This replaces the previous approach of parsing streamed graph updates.

- **Harness profiles for tool exclusions**: `backend/agents/deep/harness.py`
  - Uses `register_harness_profile(..., HarnessProfile(excluded_tools={...}))`.
  - This is the supported Deep Agents mechanism for hiding harness tools from specific models/roles.

- **Per-project FS backend**: `backend/agents/deep/workspace.py`
  - `project_filesystem_backend(project_id)` returns a `FilesystemBackend` rooted to `workspace_root/<project_id>`.

## Notes / gotchas

- **`task` tool / subagents**: Deep Agents attaches `SubAgentMiddleware` only when at least one synchronous subagent exists (including the default general-purpose subagent). To disable delegation entirely, do it via `GeneralPurposeSubagentProfile(enabled=False)` and pass no synchronous `subagents`.
- **Runtime context (`context_schema`)** exists but is still evolving in upstream; we keep project scoping explicit (workspace backend + tool argument injection where needed) rather than depending on context propagation across subagents.

