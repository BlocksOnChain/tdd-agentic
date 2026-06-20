# Graph Report - tdd-agentic  (2026-06-20)

## Corpus Check
- 130 files · ~61,324 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1633 nodes · 3294 edges · 116 communities (109 shown, 7 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 532 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e4947ebb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Ticket Models & Schemas|Ticket Models & Schemas]]
- [[_COMMUNITY_LLM Routing & RAG Pipeline|LLM Routing & RAG Pipeline]]
- [[_COMMUNITY_Event Bus & Ticket Service|Event Bus & Ticket Service]]
- [[_COMMUNITY_Frontend Components & UI|Frontend Components & UI]]
- [[_COMMUNITY_Agent Graph & Specialist Subgraphs|Agent Graph & Specialist Subgraphs]]
- [[_COMMUNITY_Project Manager & Routing|Project Manager & Routing]]
- [[_COMMUNITY_Message Reducer & Common Utils|Message Reducer & Common Utils]]
- [[_COMMUNITY_Agent Logs & EventBus Persistence|Agent Logs & EventBus Persistence]]
- [[_COMMUNITY_Frontend Dependencies|Frontend Dependencies]]
- [[_COMMUNITY_Researcher Subgraph & Code Tools|Researcher Subgraph & Code Tools]]
- [[_COMMUNITY_Skills Registry System|Skills Registry System]]
- [[_COMMUNITY_Agent API Routes & Checkpoint Cache|Agent API Routes & Checkpoint Cache]]
- [[_COMMUNITY_Frontend TypeScript Config|Frontend TypeScript Config]]
- [[_COMMUNITY_DB Session & Ticket API Routes|DB Session & Ticket API Routes]]
- [[_COMMUNITY_Checkpointer & App Lifecycle|Checkpointer & App Lifecycle]]
- [[_COMMUNITY_Observability & Stream Helpers|Observability & Stream Helpers]]
- [[_COMMUNITY_Documentation & Design Docs|Documentation & Design Docs]]
- [[_COMMUNITY_Checkpoint List Cache|Checkpoint List Cache]]
- [[_COMMUNITY_SystemState & Cancellation|SystemState & Cancellation]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 94|Community 94]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 96|Community 96]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 100|Community 100]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 102|Community 102]]
- [[_COMMUNITY_Community 103|Community 103]]
- [[_COMMUNITY_Community 104|Community 104]]
- [[_COMMUNITY_Community 105|Community 105]]
- [[_COMMUNITY_Community 106|Community 106]]
- [[_COMMUNITY_Community 107|Community 107]]
- [[_COMMUNITY_Community 108|Community 108]]
- [[_COMMUNITY_Community 109|Community 109]]
- [[_COMMUNITY_Community 110|Community 110]]
- [[_COMMUNITY_Community 112|Community 112]]
- [[_COMMUNITY_Community 113|Community 113]]
- [[_COMMUNITY_Community 114|Community 114]]
- [[_COMMUNITY_Community 115|Community 115]]
- [[_COMMUNITY_Community 116|Community 116]]

## God Nodes (most connected - your core abstractions)
1. `SubtaskStatus` - 94 edges
2. `AgentRole` - 85 edges
3. `get_settings()` - 78 edges
4. `TicketStatus` - 75 edges
5. `Event` - 70 edges
6. `SystemState` - 50 edges
7. `TodoStatus` - 49 edges
8. `build_specialist_subgraph()` - 35 edges
9. `str` - 33 edges
10. `Ticket` - 32 edges

## Surprising Connections (you probably didn't know these)
- `Agent Notes` --semantically_similar_to--> `Qdrant Vector Store`  [INFERRED] [semantically similar]
  backend/workspace_seed/AGENTS.md → docker-compose.yml
- `TicketStatus` --uses--> `TicketStatus`  [INFERRED]
  backend/tests/test_ticket_state_machine.py → backend/ticket_system/models.py
- `bool` --uses--> `TicketStatus`  [INFERRED]
  backend/tests/test_ticket_state_machine.py → backend/ticket_system/models.py
- `str` --uses--> `TicketStatus`  [INFERRED]
  backend/tests/test_ticket_tools.py → backend/ticket_system/models.py
- `TicketStatus` --uses--> `TicketStatus`  [INFERRED]
  backend/tests/test_ticket_tools.py → backend/ticket_system/models.py

## Import Cycles
- 1-file cycle: `backend/tests/test_ticket_tools.py -> backend/tests/test_ticket_tools.py`
- 1-file cycle: `backend/ticket_system/models.py -> backend/ticket_system/models.py`
- 1-file cycle: `backend/api/main.py -> backend/api/main.py`

## Communities (116 total, 7 thin omitted)

### Community 0 - "Ticket Models & Schemas"
Cohesion: 0.09
Nodes (21): 1) UI (Next.js), 2) API + event stream (FastAPI + WebSocket), 3) The “team brain” (LangGraph root graph), 4) Durable memory (PostgreSQL + Qdrant + Workspace), Channel A: checkpointed shared state (short + durable), Channel B: “external truth” (source of record), Code + test tools (how devs change the repo), Diagram: end-to-end architecture (who talks to whom) (+13 more)

### Community 1 - "LLM Routing & RAG Pipeline"
Cohesion: 0.24
Nodes (8): coordinator_model(), Parse a role slug into (gateway_provider, model_id_for_that_gateway).      Direc, Coordinator uses dev model by default (planning is done by Lead)., Coordinator uses dev model by default (planning is done by Lead)., Coordinator uses dev model by default (planning is done by Lead)., _split_slug(), test_split_slug_explicit_openrouter(), test_split_slug_known_direct_providers()

### Community 2 - "Event Bus & Ticket Service"
Cohesion: 0.22
Nodes (15): detect_agent_runtime(), get_agent_runtime_agents_md_section(), get_agent_runtime_prompt_section(), _mongodb_memory_server_hints(), _parse_os_release(), Agent execution host facts for prompts and workspace AGENTS.md.  ``shell_run`` a, Markdown section seeded into every new project AGENTS.md., Best-effort facts about the process that runs agent tools. (+7 more)

### Community 3 - "Frontend Components & UI"
Cohesion: 0.12
Nodes (14): AgentRole, CheckpointT, LogItemResponse, PersistedAgentLog, SubtaskStatus, SubtaskT, TestCaseSpec, TestType (+6 more)

### Community 4 - "Agent Graph & Specialist Subgraphs"
Cohesion: 0.09
Nodes (36): int, Path, str, Any, int, str, Researcher subgraph — web search, doc writing, RAG ingestion, skill creation., fs_delete() (+28 more)

### Community 5 - "Project Manager & Routing"
Cohesion: 0.15
Nodes (23): str, _format_pm_handoff(), _infer_next_dev_route(), _normalise_ticket_ids(), Format a handoff using the compact Handoff protocol., Format a handoff using the compact Handoff protocol., Deterministic routing to a dev role based on ticket/subtask state.      Policy:, Deterministic routing to a dev role based on ticket/subtask state.      Policy: (+15 more)

### Community 6 - "Message Reducer & Common Utils"
Cohesion: 0.11
Nodes (26): add_messages_trimmed(), Checkpoint message reducer — merge then trim human handoffs., Keep first human, recent humans, bounded AI messages.      This is the primary c, trim_checkpoint_messages(), AIMessage, int, Tests for AI message trimming in checkpoints., When AI messages exceed max_ai, keep first + tail. (+18 more)

### Community 7 - "Agent Logs & EventBus Persistence"
Cohesion: 0.13
Nodes (14): get_cached_lead_appendix(), Centralized system prompts for every agent role.  Kept in one place so they can, Return the RITE + tool contract appendix (cached once)., Return the RITE + tool contract appendix (cached once)., Return the RITE + tool contract appendix (cached once)., Return the RITE + tool contract appendix (cached once)., Return the RITE + tool contract appendix (cached once)., Return the RITE + tool contract appendix (cached once). (+6 more)

### Community 8 - "Frontend Dependencies"
Cohesion: 0.08
Nodes (24): dependencies, clsx, lucide-react, next, react, react-dom, tailwind-merge, zustand (+16 more)

### Community 9 - "Researcher Subgraph & Code Tools"
Cohesion: 0.26
Nodes (24): AgentRole, AnswerQuestion, Any, bool, int, ProjectCreate, SubtaskCreate, SubtaskStatus (+16 more)

### Community 10 - "Skills Registry System"
Cohesion: 0.11
Nodes (30): str, int, str, Any, Path, str, create_skill(), Create or update a Skill (a focused capability brief) and assign it to roles. (+22 more)

### Community 11 - "Agent API Routes & Checkpoint Cache"
Cohesion: 0.13
Nodes (21): get_file_content(), Routes for starting, resuming, and inspecting LangGraph agent runs., Resume a paused graph after the human answered an interrupt()., Resume a paused graph after the human answered an interrupt()., Resume a project's graph from its last persisted checkpoint., Resume a project's graph from its last persisted checkpoint., Resume the graph from a specific historical checkpoint., Resume the graph from a specific historical checkpoint. (+13 more)

### Community 12 - "Frontend TypeScript Config"
Cohesion: 0.10
Nodes (19): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+11 more)

### Community 13 - "DB Session & Ticket API Routes"
Cohesion: 0.31
Nodes (9): datetime, str, datetime, Base, Shared declarative base for all ORM models., DeclarativeBase, SQLAlchemy ORM models for the ticket platform.  Hierarchy: Project → Ticket → Su, _utcnow() (+1 more)

### Community 14 - "Checkpointer & App Lifecycle"
Cohesion: 0.10
Nodes (30): _backoff_seconds(), _is_provider_400_error(), Multi-provider LLM factory.  Adds two production-grade behaviours on top of plai, Wrap a Runnable: on provider-400 error, sleep 60s and retry once., Exponential backoff with jitter (attempt is 0-based)., Retry transient LLM errors with exponential backoff (429, 5xx, network)., Attach inline retry-with-backoff to a chat model.      The returned object is a, Wrap a Runnable: on provider-400 error, sleep 60s and retry once. (+22 more)

### Community 15 - "Observability & Stream Helpers"
Cohesion: 0.12
Nodes (24): Any, str, _broadcast_interrupts(), _checkpoint_id(), get_agent_log_item(), get_state(), list_project_files(), Drive the graph until it yields or completes; raises on error.      After each u (+16 more)

### Community 16 - "Documentation & Design Docs"
Cohesion: 0.16
Nodes (16): Agent Context and Workflow, Agent Hierarchy, Agent Notes, Agent Workspace Readme, CRAG Retrieval Pipeline, Docker Compose, Docker Compose Dev, Human-in-the-Loop Interface (+8 more)

### Community 17 - "Checkpoint List Cache"
Cohesion: 0.05
Nodes (56): _payload_for_storage(), persist_agent_event(), Persist agent-scoped events from the EventBus to ``agent_logs`` for replay., Return a JSON-serialisable copy of the payload., Write one ``agent`` bus event to Postgres (best-effort, non-blocking graph)., _build_serde(), close_pool(), get_pool() (+48 more)

### Community 18 - "SystemState & Cancellation"
Cohesion: 0.15
Nodes (15): float, cancel_all_running_tasks(), _cancel_task(), Cancel a single task and wait for it to settle. Returns True if it stopped in ti, Cancel a single task and wait for it to settle. Returns True if it stopped in ti, Cancel any in-flight graph execution for the given project.      Waits up to ~10, Cancel any in-flight graph execution for the given project.      Waits up to ~10, Await a task, suppressing CancelledError + any final exception. (+7 more)

### Community 19 - "Community 19"
Cohesion: 0.12
Nodes (34): grader_model(), researcher_model(), Any, int, str, Any, float, int (+26 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (26): Classify the turn's verification signal from ``run_tests`` results.      Returns, Classify the turn's verification signal from ``run_tests`` results.      Returns, Classify the turn's verification signal from ``run_tests`` results.      Returns, IDs of subtasks the agent flipped to ``done`` via update_subtask_status., IDs of subtasks the agent flipped to ``done`` via update_subtask_status., IDs of subtasks the agent flipped to ``done`` via update_subtask_status., Classify the turn's verification signal from ``run_tests`` results.      Returns, IDs of subtasks the agent flipped to ``done`` via update_subtask_status. (+18 more)

### Community 21 - "Community 21"
Cohesion: 0.13
Nodes (14): list_agent_logs(), Read historical agent logs from Postgres., Return newest-first rows; caller may reverse for chronological UI., Any, AsyncSession, int, str, int (+6 more)

### Community 22 - "Community 22"
Cohesion: 0.33
Nodes (5): Connection, do_run_migrations(), Alembic env.py — async-aware migration runner., run_async_migrations(), run_migrations_online()

### Community 23 - "Community 23"
Cohesion: 0.16
Nodes (22): _engaged_subtask_id_from_tools(), _latest_subtask_id_from_tools(), Subtask the agent picked up via ``next_pending*`` this turn (``None`` = no work), IDs of subtasks the agent flipped to ``blocked`` via update_subtask_status., Subtask the agent picked up via ``next_pending*`` this turn (``None`` = no work), IDs of subtasks the agent flipped to ``blocked`` via update_subtask_status., Classify whether the agent resolved the subtask it picked up.      Returns ``(st, _subtask_resolution_outcome() (+14 more)

### Community 24 - "Community 24"
Cohesion: 0.05
Nodes (38): 1. Tool Calling Analysis, 1A. Current tool definition quality, 1B. Tool calling patterns across agents, 1C. Tool error handling, 2. System Prompt Analysis, 2A. Current prompt structure, 2B. Prompt engineering patterns currently used, 2C. Modern prompt engineering patterns NOT used (+30 more)

### Community 27 - "Community 27"
Cohesion: 0.10
Nodes (48): AsyncSession, str, Ticket, Subtask, add_question(), all_subtasks_done(), answer_question(), create_ticket() (+40 more)

### Community 38 - "Community 38"
Cohesion: 0.50
Nodes (4): callbacks_for(), get_langfuse_handler(), Optional Langfuse callback handler for LangGraph traces.  Returns ``None`` when, str

### Community 39 - "Community 39"
Cohesion: 0.17
Nodes (16): str, test_parse_dev_role_rejects_non_dev_roles(), next_pending_subtask(), next_pending_subtask_in_project(), _parse_dev_role(), Return the next pending subtask (lowest order_index) for a ticket, optionally fi, Parse an optional dev role; treat empty / 'none' / 'null' as unfiltered., Return the next pending subtask for a role across the whole project.      Uses t (+8 more)

### Community 51 - "Community 51"
Cohesion: 0.10
Nodes (25): ContextEntry, ContextStore, Transient context store for inter-agent communication.  Agents write compact out, In-memory store for agent context references.      Each entry is a pointer, not, Add a context entry, return its ID., Look up a context entry by reference ID., Return all entries referencing a specific ticket., Return all entries from a specific agent. (+17 more)

### Community 52 - "Community 52"
Cohesion: 0.07
Nodes (32): Handoff, HandoffV2, Phase, Structured inter-agent handoff protocol.  Replaces the natural-language handoff, Create a Handoff from a RoutingDecision., Extended handoff that carries a small ContextStore snapshot.      Instead of ful, Best-effort map an arbitrary phase string to a valid Phase.          The PM mode, Compact structured handoff between agents. (+24 more)

### Community 53 - "Community 53"
Cohesion: 0.17
Nodes (20): _agent_from_wrote_nodes(), _cache_bytes_total(), _estimate_bytes(), _evict_cache_if_needed(), _fetch_checkpoints(), get_checkpoints_list(), invalidate_checkpoint_list_cache(), _lock() (+12 more)

### Community 54 - "Community 54"
Cohesion: 0.10
Nodes (19): 1A. Trim AI messages on checkpoint write, 1B. Remove dead SystemState fields, 1C. Single canonical project goal, 2A. Compact handoff format, 2B. Smart context lookup via context_refs, 2C. Agent-to-agent direct handoff, 3A. Static system prompt cache, 3B. Skill loading optimization (+11 more)

### Community 55 - "Community 55"
Cohesion: 0.17
Nodes (14): _last_text(), Best-effort textual summary of the final AIMessage in ``messages``., Best-effort textual summary of the final AIMessage in ``messages``., ExecutionPlan, Root LangGraph state for the multi-agent system.  Important: everything in this, A single RITE-format test case in a plan., A single subtask in an execution plan (Lead's output)., Execution plan produced by Lead agent, consumed by Coordinator. (+6 more)

### Community 56 - "Community 56"
Cohesion: 0.14
Nodes (14): get_cached_role_base(), Return the static base for a given role (cached once).      Only called at app s, Return the static base for a given role (cached once).      Only called at app s, Return the static base for a given role (cached once).      Only called at app s, Return the static base for a given role (cached once).      Only called at app s, Return the static base for a given role (cached once).      Only called at app s, Return the static base for a given role (cached once).      Only called at app s, Return the static base for a given role (cached once).      Only called at app s (+6 more)

### Community 57 - "Community 57"
Cohesion: 0.12
Nodes (16): Agent workflow and context, Already in the design, Checkpointed state (`SystemState`), Efficiency and reducing context, External persistence, Highest-impact improvements, How context is carried between steps, Key source files (+8 more)

### Community 58 - "Community 58"
Cohesion: 0.12
Nodes (15): 1. Configure, 2. Boot the stack, 3. Use it, Agent hierarchy, API endpoints, Architecture, How TDD is enforced, Local development without Docker (+7 more)

### Community 59 - "Community 59"
Cohesion: 0.08
Nodes (16): PM step 5 must review subtask count per ticket., PM step 5 must review subtask count per ticket., Backend lead must have minimum subtask count constraint., Merged lead must have minimum subtask count constraint and decomposition guidanc, Frontend lead must have minimum subtask count constraint., _ticket_ready_for_todo must enforce minimum subtask count (via source inspection, _ticket_ready_for_todo must enforce minimum subtask count (via source inspection, Verify the PM and lead prompts force fine-grained decomposition. (+8 more)

### Community 60 - "Community 60"
Cohesion: 0.16
Nodes (19): AgentEvent, emit(), _format_tool_error(), last_ai_message(), Shared utilities for agent subgraphs.  Provides helpers for invoking a tool-call, Persist an event to the EventBus and return an AgentEvent for state., Convert an exception (often a Pydantic ValidationError) into an     actionable c, Execute tool calls in an AI message and return the resulting ToolMessages. (+11 more)

### Community 61 - "Community 61"
Cohesion: 0.14
Nodes (48): AgentEvent, AgentRole, AnswerQuestion, SubtaskCreate, SubtaskStatus, SubtaskUpdate, TicketCreate, TicketUpdate (+40 more)

### Community 62 - "Community 62"
Cohesion: 0.11
Nodes (14): Verify dev prompts include test failure diagnosis instructions., Dev system prompt must have a TEST FAILURE HANDLING section., Verify dev prompts include test failure diagnosis instructions., Dev system prompt must have a TEST FAILURE HANDLING section., Dev prompt must instruct using shell_run for diagnosis., Dev prompt must instruct using shell_run for diagnosis., Dev prompt must stop after 3 identical failures., Dev prompt must stop after 3 identical failures. (+6 more)

### Community 63 - "Community 63"
Cohesion: 0.16
Nodes (11): AgentLog(), AGENT_COLORS, AgentActivityPanel(), TicketProgress(), CheckpointsPanel(), CrashBanner(), ProjectPicker(), api (+3 more)

### Community 64 - "Community 64"
Cohesion: 0.14
Nodes (8): Verify the PM and lead prompts mandate infrastructure scaffolding., PM must dispatch TWO agents in parallel at step 1., PM must not move ticket to TODO without infrastructure., Lead must create infrastructure subtask at order_index 0 via devops., Merged lead must cover both backend and frontend infrastructure., DevOps system prompt must list specific scaffolding deliverables., DevOps system prompt must list specific scaffolding deliverables., TestBug1InfrastructureScaffolding

### Community 65 - "Community 65"
Cohesion: 0.12
Nodes (23): DetailModal(), formatAbsoluteTime(), formatRelativeTime(), LogRow(), AGENT_LABELS, AGENT_TEXT, agentLabel(), agentPillClasses() (+15 more)

### Community 66 - "Community 66"
Cohesion: 0.27
Nodes (7): FILE_ICONS, FileExplorer(), getColorForExt(), getIcon(), Node, TreeRoot(), cn()

### Community 67 - "Community 67"
Cohesion: 0.39
Nodes (11): _cuda_major_from_nvidia_smi(), _has_nvidia_gpu(), install_torch(), main(), project_extras(), bool, str, Return one of: skip, pypi, cpu, cu124, cu118. (+3 more)

### Community 68 - "Community 68"
Cohesion: 0.24
Nodes (16): MonkeyPatch, _fallback_routing_decision(), Pick backend_lead or frontend_lead from DB state when the LLM routing fails., Pick backend_lead or frontend_lead from DB state when the LLM routing fails., Pick lead from DB state when the LLM routing fails.      This keeps the graph al, SimpleNamespace, test_split_slug_openrouter_vendor_model_when_key_set(), DB-style fallback when the PM model omits valid routing JSON or chooses end too (+8 more)

### Community 69 - "Community 69"
Cohesion: 0.23
Nodes (12): bool, Ticket, _advance_in_review_to_todo(), Conservative readiness check for moving IN_REVIEW -> TODO.      User expectation, Conservative readiness check for moving IN_REVIEW -> TODO.      User expectation, Auto-advance tickets from IN_REVIEW to TODO when they are ready.      Returns th, Conservative readiness check for moving IN_REVIEW -> TODO.      User expectation, Auto-advance tickets from IN_REVIEW to TODO when they are ready.      Returns th (+4 more)

### Community 70 - "Community 70"
Cohesion: 0.25
Nodes (10): _parse_routing(), Best-effort JSON extraction from the supervisor's final message., Best-effort JSON extraction from the supervisor's final message., Best-effort JSON extraction from the supervisor's final message., test_parses_ticket_ids_and_phase(), Verify the PM supervisor's JSON routing parser tolerates common LLM outputs., test_parses_json_in_markdown_fence(), test_parses_optional_ticket_ids() (+2 more)

### Community 71 - "Community 71"
Cohesion: 0.20
Nodes (9): Tests for skill loader caching and change detection., Calling inject_skills twice with same skill set returns unchanged base., When a role has no skills, inject_skills returns base unchanged., When skill set changes, injection happens and cache updates., Different roles have separate caches., test_inject_cache_per_role(), test_inject_skills_injects_when_new_skills(), test_inject_skills_returns_base_when_no_skills() (+1 more)

### Community 72 - "Community 72"
Cohesion: 0.29
Nodes (6): metadata, EventBridge(), Shell(), persistedLogToEntry(), useEventStream(), WSEvent

### Community 73 - "Community 73"
Cohesion: 0.17
Nodes (4): Tests for the three bug fixes.  Bug 1: PM/leads must create infrastructure subta, Ensure the fixes don't break existing prompt behavior., Ensure the fixes don't break existing prompt behavior., TestRegressionNoRegresion

### Community 74 - "Community 74"
Cohesion: 0.14
Nodes (21): _build_specialist_input(), Assemble the minimal message list a specialist needs to act.      Strategy:, Assemble the minimal message list a specialist needs to act.      Strategy:, Assemble the minimal message list a specialist needs to act.      Strategy:, SystemState, str, SystemState, _make_state() (+13 more)

### Community 75 - "Community 75"
Cohesion: 0.29
Nodes (4): DEFAULT_COLORS, HitlPanel(), InterruptItem, KIND_COLORS

### Community 76 - "Community 76"
Cohesion: 0.16
Nodes (16): Root state object shared across the orchestration graph., Root state object shared across the orchestration graph., SystemState, Tests for SystemState changes (dead fields neutralized, context_store added)., context_store defaults to an empty dict., context_store can be set to a dict., project_context exists and can hold the project goal., active_ticket_id and active_subtask_id exist. (+8 more)

### Community 77 - "Community 77"
Cohesion: 0.18
Nodes (16): AgentRole, int, SimpleNamespace, str, SubtaskStatus, Unit tests for actionable subtask selection helpers., _sub(), test_first_actionable_subtask_ignores_done() (+8 more)

### Community 78 - "Community 78"
Cohesion: 0.25
Nodes (8): HitlBadge(), NAV, InterruptPanel(), AgentLogEntry, TicketT, CrashState, PendingInterrupt, UIState

### Community 79 - "Community 79"
Cohesion: 0.10
Nodes (22): build_lead_subgraph(), Root LangGraph orchestration graph.  The Project Manager acts as the supervisor;, Conditional edge: dispatch from the project manager based on its decision., Conditional edge: dispatch from the project manager based on its decision., Build the merged Lead agent subgraph (handles both backend and frontend planning, _route_from_pm(), lead_model(), build_specialist_subgraph() (+14 more)

### Community 80 - "Community 80"
Cohesion: 0.29
Nodes (6): Agent execution environment, Agent notes (this workspace), Framework stacks (e.g. Next.js), Native binaries vs npm, What not to do, Where documentation lives

### Community 81 - "Community 81"
Cohesion: 0.33
Nodes (6): int, Embeddings, embedding_dim(), Embedding model factory — OpenAI by default, local sentence-transformers as fall, Return the embedding vector size for the active provider., Return the embedding vector size for the active provider.

### Community 84 - "Community 84"
Cohesion: 0.33
Nodes (11): describe_llm_slug(), log_llm_invoke_exception_context(), log_llm_invoke_start(), log_rag_crag_llm_targets(), Structured logging around LLM HTTP calls — see where traffic is sent., Return (provider, model_name, human_readable_endpoint_hint)., Call from inside an ``except`` block — attaches stack trace via ``exception()``., Call from inside an ``except`` block — attaches stack trace via ``exception()``. (+3 more)

### Community 85 - "Community 85"
Cohesion: 0.33
Nodes (4): str, Application configuration loaded from environment via pydantic-settings., Settings, BaseSettings

### Community 86 - "Community 86"
Cohesion: 0.12
Nodes (26): _build_specialist_system_prompt(), _extract_json_object(), _format_structured_summary(), _parse_next_pending_tool_result(), _parse_structured_summary(), Generic tool-using agent runner shared by specialist subgraphs.  Each specialist, Parse researcher structured output or return ``None`` for prose fallback., Render structured researcher output as a compact hand-off summary. (+18 more)

### Community 87 - "Community 87"
Cohesion: 0.26
Nodes (12): Any, str, complete_assignment(), _dump(), Persistence tools for Coordinator agent.  These tools handle all database writes, Mark a subtask as done and transition related ticket if ready.      USE WHEN: De, Create or update a ticket in the database.      USE WHEN: Persisting a new ticke, Transition a ticket to a new status.      USE WHEN: Moving tickets through the w (+4 more)

### Community 88 - "Community 88"
Cohesion: 0.14
Nodes (13): Tests for prompt caching utility., The lead appendix contains RITE contract and tool contract., Calling get_cached_role_base multiple times returns the same object., The PM prompt includes a routing protocol section., The PM prompt includes a constraints section., The PM prompt includes a tool selection guide., get_cached_role_base returns the correct prompt for each role., test_cached_lead_appendix_not_empty() (+5 more)

### Community 89 - "Community 89"
Cohesion: 0.40
Nodes (4): str, ask_human(), Human-in-the-loop tool that pauses graph execution via ``interrupt()``.  Calling, Ask the human supervisor a question and wait for the answer.      Use sparingly

### Community 90 - "Community 90"
Cohesion: 0.17
Nodes (11): 10. Integration Back into tdd-agentic, 11. Decision Matrix (Quick Reference), 12. Open Research Questions (for follow-up agent), 13. Key Repository Files, 14. External References, 15. Suggested Deliverables from Research Agent, 1. Executive Summary, 2. What “Efficiency” Means in This Project (+3 more)

### Community 91 - "Community 91"
Cohesion: 0.19
Nodes (15): datetime, int, SimpleNamespace, str, TicketStatus, Tests for PM ticket tool helpers and list_tickets etag caching., test_list_tickets_returns_etag_and_tickets(), test_list_tickets_returns_full_payload_when_etag_stale() (+7 more)

### Community 92 - "Community 92"
Cohesion: 0.18
Nodes (11): 5. Option Taxonomy (All Paths), Option A — No fine-tuning (baseline & data collection), Option B — Prompt / skill distillation (soft training), Option C — LoRA SFT on Apple Silicon (MLX), Option D — Cloud GPU training (HF Jobs / Unsloth CUDA) → GGUF local, Option E — Per-role adapter suite (recommended training architecture), Option F — Distillation from cloud teachers, Option G — Synthetic dataset generation (+3 more)

### Community 94 - "Community 94"
Cohesion: 0.24
Nodes (12): int, MonkeyPatch, str, Exception, DummyRunnable, Provider400Error, Minimal Runnable-like object for retry wrapper tests., test_generic_400_does_not_sleep_or_retry() (+4 more)

### Community 95 - "Community 95"
Cohesion: 0.18
Nodes (12): get_checkpointer(), Async context manager that yields an initialized AsyncPostgresSaver.      On fir, build_root_graph(), Compile the full multi-agent orchestration graph., Compile the full multi-agent orchestration graph., BaseCheckpointSaver, list_interrupts(), Branch off and resume the graph from a specific historical checkpoint. (+4 more)

### Community 96 - "Community 96"
Cohesion: 0.33
Nodes (5): bool, TicketStatus, Smoke tests for the ticket-system state-machine logic.  These exercise the pure-, test_transition_allowed(), TicketStatus

### Community 97 - "Community 97"
Cohesion: 0.20
Nodes (10): 6.1 Training formats, 6.2 Per-role dataset contents, 6.3 Trace collection hooks (repo), 6.4 Data quality filters, 6. Dataset Design (Project-Specific), Coordinator (`tdd-coordinator`), Dev (`tdd-dev-tdd`), Lead (`tdd-lead-planning`) (+2 more)

### Community 98 - "Community 98"
Cohesion: 0.29
Nodes (9): _parse_execution_plan(), Parse Lead's final message into an ``ExecutionPlan`` (Coordinator reads state)., ExecutionPlan, Tests for Lead execution_plan parsing in runner., order_index is not part of SubtaskPlan — must not break validation., test_parse_execution_plan_markdown_fence(), test_parse_execution_plan_plain_json(), test_parse_execution_plan_returns_none_on_garbage() (+1 more)

### Community 99 - "Community 99"
Cohesion: 0.25
Nodes (8): 9. Recommended Phased Plan, Phase 0 — Baseline (1–2 days), Phase 1 — Prompt/tool hardening (no train), Phase 2 — Trace collection (1–2 weeks parallel use), Phase 3 — First LoRA (Mac), Phase 4 — PM + Lead adapters, Phase 5 — Optional cloud pass, Phase 6 — RAG embedding tune (optional)

### Community 100 - "Community 100"
Cohesion: 0.33
Nodes (6): 3.1 Agent roles and LLM slots, 3.2 Tool inventory by agent, 3.3 Skills system, 3.4 Prompts and structured outputs, 3.5 Existing local integration points, 3. Project Architecture (Training-Relevant)

### Community 101 - "Community 101"
Cohesion: 0.10
Nodes (25): add_todo_to_subtask(), create_subtask(), _dump(), LangChain tools that wrap the ticket-system service for use by agents.  These to, A single RITE-format test case the lead must specify per subtask., A single RITE-format test case the lead must specify per subtask., Create an ordered subtask under a ticket using RITE-format test cases.      Idem, A single RITE-format test case the lead must specify per subtask. (+17 more)

### Community 102 - "Community 102"
Cohesion: 0.53
Nodes (3): object, _normalise_test_cases(), Accept either plain strings (legacy) or structured dicts and     return a list o

### Community 103 - "Community 103"
Cohesion: 0.13
Nodes (18): pm_model(), int, SystemState, build_project_manager_node(), _build_system_prompt(), _condense_messages_for_supervisor(), _infer_fallback_route(), Project Manager — supervisor agent.  Acts as the routing decision-maker. Holds t (+10 more)

### Community 104 - "Community 104"
Cohesion: 0.67
Nodes (3): Subtask order_index idempotency helpers., test_normalise_title_strips_edges(), _normalise_title()

### Community 105 - "Community 105"
Cohesion: 0.19
Nodes (8): Event, EventBus, In-memory pub/sub for realtime agent events broadcast over WebSockets.  This is, Any, str, BaseException, bool, _is_transient()

### Community 106 - "Community 106"
Cohesion: 0.67
Nodes (3): 4.1 M2 Max 64 GB constraints, 4.2 Recommended model assignment (inference), 4. Hardware Profile & Model Fit

### Community 107 - "Community 107"
Cohesion: 0.22
Nodes (9): list_subtasks(), mark_todo_done(), Mark a todo as completed., Mark a todo as completed., Mark a todo as completed., Mark a todo as completed., List subtasks filtered by project_id and optionally ticket_id and/or assigned_to, List subtasks filtered by project_id and optionally ticket_id and/or assigned_to (+1 more)

### Community 108 - "Community 108"
Cohesion: 0.10
Nodes (32): backend_dev_model(), dev_model(), devops_model(), frontend_dev_model(), get_chat_model(), _limiter_for(), qa_model(), Resolve a ``provider/model`` slug to a rate-limited chat model.      Inline retr (+24 more)

### Community 109 - "Community 109"
Cohesion: 0.22
Nodes (9): get_ticket(), get_ticket_summary(), Fetch one ticket by UUID.      USE WHEN: You need to read subtasks or RITE test, Fetch one ticket's summary — no subtask trees, no full RITE specs.      USE WHEN, Fetch one ticket's summary — no subtask trees, no full RITE specs.      USE WHEN, Fetch one ticket. ``detail='summary'`` omits RITE trees; use ``full`` for specs., Fetch one ticket by UUID.      USE WHEN: You need to read subtasks or RITE test, Fetch one ticket by UUID.      USE WHEN: You need to read subtasks or RITE test (+1 more)

### Community 110 - "Community 110"
Cohesion: 0.16
Nodes (17): _expand_query(), Expand short queries (<=2 words) with common synonyms for better recall.      Lo, Tests for RAG query expansion logic., Two-word queries are expanded with synonyms for each word., Queries with more than 2 words are returned as-is., Empty or whitespace queries handled gracefully., Unknown words produce only the original query., Each synonym adds an expanded query variant. (+9 more)

### Community 112 - "Community 112"
Cohesion: 0.40
Nodes (5): add_question_to_ticket(), Append a clarification question to a ticket and mark it questions_pending., Append a clarification question to a ticket and mark it questions_pending., Append a clarification question to a ticket and mark it questions_pending., Append a clarification question to a ticket and mark it questions_pending.

### Community 113 - "Community 113"
Cohesion: 0.40
Nodes (5): create_ticket(), Create a new ticket in DRAFT status owned by the given project.      USE WHEN: Y, Create a new ticket in DRAFT status owned by the given project.      USE WHEN: Y, Create a new ticket in DRAFT status owned by the given project.      USE WHEN: Y, Create a new ticket in DRAFT status owned by the given project.

### Community 114 - "Community 114"
Cohesion: 0.40
Nodes (5): delete_subtask(), Delete a subtask that's wrong or no longer needed. Lead-only tool.      Refuses, Delete a subtask that's wrong or no longer needed. Lead-only tool.      Refuses, Delete a subtask that's wrong or no longer needed. Lead-only tool.      Refuses, Delete a subtask that's wrong or no longer needed. Lead-only tool.      Refuses

### Community 115 - "Community 115"
Cohesion: 0.40
Nodes (5): Transition a ticket to a new status. Valid: draft, in_review, questions_pending,, Transition a ticket to a new status. Valid: draft, in_review, questions_pending,, Transition a ticket to a new status. Valid: draft, in_review, questions_pending,, Transition a ticket to a new status. Valid: draft, in_review, questions_pending,, update_ticket_status()

### Community 116 - "Community 116"
Cohesion: 0.50
Nodes (4): list_tickets(), List every ticket in a project as compact rows (id, title, status).      USE WHE, List every ticket in a project as compact rows (id, title, status).      USE WHE, List every ticket in a project as compact rows (id, title, status).      USE WHE

## Knowledge Gaps
- **236 isolated node(s):** `nextConfig`, `config`, `name`, `version`, `private` (+231 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_settings()` connect `Community 108` to `LLM Routing & RAG Pipeline`, `Agent Graph & Specialist Subgraphs`, `Message Reducer & Common Utils`, `Community 103`, `Community 38`, `Skills Registry System`, `Agent API Routes & Checkpoint Cache`, `Checkpointer & App Lifecycle`, `Community 79`, `Observability & Stream Helpers`, `Checkpoint List Cache`, `Community 81`, `Community 19`, `Community 84`, `Community 53`, `Community 85`, `Community 22`, `Community 27`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `get_cached_role_base()` connect `Community 56` to `Community 88`, `Community 73`, `Community 39`, `Agent Logs & EventBus Persistence`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `SubtaskStatus` connect `Community 61` to `Community 98`, `Community 68`, `Project Manager & Routing`, `Community 69`, `Community 103`, `Community 101`, `Researcher Subgraph & Code Tools`, `Community 74`, `Community 39`, `Community 102`, `Community 77`, `DB Session & Ticket API Routes`, `Checkpoint List Cache`, `Community 23`, `Community 86`, `Community 55`, `Community 87`, `Community 27`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._
- **Are the 84 inferred relationships involving `SubtaskStatus` (e.g. with `AgentEvent` and `bool`) actually correct?**
  _`SubtaskStatus` has 84 INFERRED edges - model-reasoned connections that need verification._
- **Are the 74 inferred relationships involving `AgentRole` (e.g. with `AgentEvent` and `bool`) actually correct?**
  _`AgentRole` has 74 INFERRED edges - model-reasoned connections that need verification._
- **Are the 65 inferred relationships involving `TicketStatus` (e.g. with `AgentEvent` and `bool`) actually correct?**
  _`TicketStatus` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 47 inferred relationships involving `Event` (e.g. with `AgentEvent` and `AIMessage`) actually correct?**
  _`Event` has 47 INFERRED edges - model-reasoned connections that need verification._