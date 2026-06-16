# Graph Report - tdd-agentic  (2026-06-16)

## Corpus Check
- 126 files · ~55,342 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1467 nodes · 2995 edges · 108 communities (99 shown, 9 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 483 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e4b737f0`
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

## God Nodes (most connected - your core abstractions)
1. `SubtaskStatus` - 91 edges
2. `AgentRole` - 83 edges
3. `Event` - 69 edges
4. `get_settings()` - 68 edges
5. `TicketStatus` - 67 edges
6. `TodoStatus` - 49 edges
7. `SystemState` - 48 edges
8. `build_specialist_subgraph()` - 34 edges
9. `str` - 33 edges
10. `Ticket` - 31 edges

## Surprising Connections (you probably didn't know these)
- `Agent Notes` --semantically_similar_to--> `Qdrant Vector Store`  [INFERRED] [semantically similar]
  backend/workspace_seed/AGENTS.md → docker-compose.yml
- `TicketStatus` --uses--> `TicketStatus`  [INFERRED]
  backend/tests/test_ticket_state_machine.py → backend/ticket_system/models.py
- `bool` --uses--> `TicketStatus`  [INFERRED]
  backend/tests/test_ticket_state_machine.py → backend/ticket_system/models.py
- `SystemState` --uses--> `SystemState`  [INFERRED]
  backend/agents/graph.py → backend/agents/state.py
- `str` --uses--> `SystemState`  [INFERRED]
  backend/agents/graph.py → backend/agents/state.py

## Import Cycles
- 1-file cycle: `backend/ticket_system/models.py -> backend/ticket_system/models.py`
- 1-file cycle: `backend/api/main.py -> backend/api/main.py`

## Communities (108 total, 9 thin omitted)

### Community 0 - "Ticket Models & Schemas"
Cohesion: 0.20
Nodes (11): _dump(), get_ticket(), get_ticket_summary(), list_tickets(), Fetch one ticket's summary — no subtask trees, no full RITE specs.      USE WHEN, List every ticket in a project as compact rows (id, title, status).      USE WHE, Fetch one ticket. ``detail='summary'`` omits RITE trees; use ``full`` for specs., Fetch one ticket by UUID.      USE WHEN: You need to read subtasks or RITE test (+3 more)

### Community 1 - "LLM Routing & RAG Pipeline"
Cohesion: 0.14
Nodes (29): backend_dev_model(), _backoff_seconds(), coordinator_model(), dev_model(), frontend_dev_model(), get_chat_model(), grader_model(), lead_model() (+21 more)

### Community 2 - "Event Bus & Ticket Service"
Cohesion: 0.22
Nodes (10): int, SystemState, _build_system_prompt(), _condense_messages_for_supervisor(), Build the PM's system prompt with XML-style context separation.      Separating, Build the PM's system prompt with XML-style context separation.      Separating, Drop ALL specialist AI/tool messages and keep only human turns.      The supervi, Drop ALL specialist AI/tool messages and keep only human turns.      The supervi (+2 more)

### Community 3 - "Frontend Components & UI"
Cohesion: 0.12
Nodes (14): AgentRole, CheckpointT, LogItemResponse, PersistedAgentLog, SubtaskStatus, SubtaskT, TestCaseSpec, TestType (+6 more)

### Community 4 - "Agent Graph & Specialist Subgraphs"
Cohesion: 0.09
Nodes (35): int, Path, str, str, Any, int, str, Researcher subgraph — web search, doc writing, RAG ingestion, skill creation. (+27 more)

### Community 5 - "Project Manager & Routing"
Cohesion: 0.17
Nodes (19): str, _advance_in_review_to_todo(), _format_pm_handoff(), _infer_fallback_route(), _infer_next_dev_route(), _normalise_ticket_ids(), Project Manager — supervisor agent.  Acts as the routing decision-maker. Holds t, Format a handoff using the compact Handoff protocol. (+11 more)

### Community 6 - "Message Reducer & Common Utils"
Cohesion: 0.06
Nodes (51): AgentEvent, emit(), _format_tool_error(), last_ai_message(), Shared utilities for agent subgraphs.  Provides helpers for invoking a tool-call, Persist an event to the EventBus and return an AgentEvent for state., Convert an exception (often a Pydantic ValidationError) into an     actionable c, Execute tool calls in an AI message and return the resulting ToolMessages. (+43 more)

### Community 7 - "Agent Logs & EventBus Persistence"
Cohesion: 0.27
Nodes (23): AgentRole, AnswerQuestion, Any, bool, int, ProjectCreate, SubtaskCreate, SubtaskStatus (+15 more)

### Community 8 - "Frontend Dependencies"
Cohesion: 0.08
Nodes (24): dependencies, clsx, lucide-react, next, react, react-dom, tailwind-merge, zustand (+16 more)

### Community 9 - "Researcher Subgraph & Code Tools"
Cohesion: 0.16
Nodes (19): int, Any, int, str, Embeddings, embedding_dim(), get_embeddings(), Embedding model factory — OpenAI by default, local sentence-transformers as fall (+11 more)

### Community 10 - "Skills Registry System"
Cohesion: 0.11
Nodes (30): str, int, str, Any, Path, str, create_skill(), Create or update a Skill (a focused capability brief) and assign it to roles. (+22 more)

### Community 11 - "Agent API Routes & Checkpoint Cache"
Cohesion: 0.13
Nodes (23): invalidate_checkpoint_list_cache(), Drop cached checkpoint lists after agent actions or globally., Drop cached checkpoint lists after agent actions or globally., Routes for starting, resuming, and inspecting LangGraph agent runs., Resume a project's graph from its last persisted checkpoint., Resume a project's graph from its last persisted checkpoint., Cancel any in-flight graph execution for the given project.      Waits up to ~10, Cancel any in-flight graph execution for the given project.      Waits up to ~10 (+15 more)

### Community 12 - "Frontend TypeScript Config"
Cohesion: 0.10
Nodes (19): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+11 more)

### Community 13 - "DB Session & Ticket API Routes"
Cohesion: 0.15
Nodes (15): _payload_for_storage(), persist_agent_event(), Persist agent-scoped events from the EventBus to ``agent_logs`` for replay., Return a JSON-serialisable copy of the payload., Write one ``agent`` bus event to Postgres (best-effort, non-blocking graph)., Event, EventBus, In-memory pub/sub for realtime agent events broadcast over WebSockets.  This is (+7 more)

### Community 14 - "Checkpointer & App Lifecycle"
Cohesion: 0.14
Nodes (17): Wrap a Runnable: on provider-400 error, sleep 60s and retry once., Retry transient LLM errors with exponential backoff (429, 5xx, network)., Attach inline retry-with-backoff to a chat model.      The returned object is a, Wrap a Runnable: on provider-400 error, sleep 60s and retry once., Parse a role slug into (gateway_provider, model_id_for_that_gateway).      Direc, Public alias so callers can wrap a tool-bound model in retry logic., Public alias so callers can wrap a tool-bound model in retry logic., Attach inline retry-with-backoff to a chat model or tool-bound Runnable. (+9 more)

### Community 15 - "Observability & Stream Helpers"
Cohesion: 0.11
Nodes (25): Any, str, _broadcast_interrupts(), _checkpoint_id(), get_agent_log_item(), get_file_content(), list_project_files(), Drive the graph until it yields or completes; raises on error.      After each u (+17 more)

### Community 16 - "Documentation & Design Docs"
Cohesion: 0.16
Nodes (16): Agent Context and Workflow, Agent Hierarchy, Agent Notes, Agent Workspace Readme, CRAG Retrieval Pipeline, Docker Compose, Docker Compose Dev, Human-in-the-Loop Interface (+8 more)

### Community 17 - "Checkpoint List Cache"
Cohesion: 0.20
Nodes (17): AsyncSession, str, AsyncSession, get_db(), Async SQLAlchemy engine, session factory, and DeclarativeBase., FastAPI dependency that yields a session and ensures cleanup., answer(), create_subtask() (+9 more)

### Community 18 - "SystemState & Cancellation"
Cohesion: 0.22
Nodes (11): float, cancel_all_running_tasks(), _cancel_task(), Cancel a single task and wait for it to settle. Returns True if it stopped in ti, Cancel a single task and wait for it to settle. Returns True if it stopped in ti, Await a task, suppressing CancelledError + any final exception., Await a task, suppressing CancelledError + any final exception., Cancel every in-flight agent run. Used by lifespan shutdown so the     process c (+3 more)

### Community 19 - "Community 19"
Cohesion: 0.10
Nodes (36): Any, float, int, str, Document, _cache_get(), _cache_put(), crag_retrieve() (+28 more)

### Community 20 - "Community 20"
Cohesion: 0.16
Nodes (20): Classify the turn's verification signal from ``run_tests`` results.      Returns, Classify the turn's verification signal from ``run_tests`` results.      Returns, Classify the turn's verification signal from ``run_tests`` results.      Returns, _verification_outcome(), str, ToolMessage, Tests for the devops/infra completion verification gate.  Regression: devops rep, Scaffolding files without running a test does not prove the infra works. (+12 more)

### Community 21 - "Community 21"
Cohesion: 0.11
Nodes (17): list_agent_logs(), Read historical agent logs from Postgres., Return newest-first rows; caller may reverse for chronological UI., Any, AsyncSession, int, str, BaseException (+9 more)

### Community 22 - "Community 22"
Cohesion: 0.33
Nodes (5): Connection, do_run_migrations(), Alembic env.py — async-aware migration runner., run_async_migrations(), run_migrations_online()

### Community 23 - "Community 23"
Cohesion: 0.27
Nodes (13): _subtask_resolution_outcome(), str, ToolMessage, Tests for the dev-agent subtask resolution gate.  Regression: backend_dev read f, test_engaged_subtask_from_next_pending_in_project(), test_engaged_subtask_null_means_no_work(), test_resolution_incomplete_when_subtask_not_closed(), test_resolution_no_engagement_without_subtask_id() (+5 more)

### Community 24 - "Community 24"
Cohesion: 0.05
Nodes (38): 1. Tool Calling Analysis, 1A. Current tool definition quality, 1B. Tool calling patterns across agents, 1C. Tool error handling, 2. System Prompt Analysis, 2A. Current prompt structure, 2B. Prompt engineering patterns currently used, 2C. Modern prompt engineering patterns NOT used (+30 more)

### Community 27 - "Community 27"
Cohesion: 0.09
Nodes (52): AsyncSession, str, Ticket, Subtask, Subtask order_index idempotency helpers., test_normalise_title_strips_edges(), add_question(), all_subtasks_done() (+44 more)

### Community 38 - "Community 38"
Cohesion: 0.29
Nodes (7): callbacks_for(), get_langfuse_handler(), Optional Langfuse callback handler for LangGraph traces.  Returns ``None`` when, str, Branch off and resume the graph from a specific historical checkpoint., Branch off and resume the graph from a specific historical checkpoint., _resume_from_checkpoint()

### Community 39 - "Community 39"
Cohesion: 0.24
Nodes (12): str, next_pending_subtask(), next_pending_subtask_in_project(), _parse_dev_role(), Return the next pending subtask (lowest order_index) for a ticket, optionally fi, Parse an optional dev role; treat empty / 'none' / 'null' as unfiltered., Return the next pending subtask for a role across the whole project.      Uses t, Return the next pending subtask for a role across the whole project.      USE WH (+4 more)

### Community 51 - "Community 51"
Cohesion: 0.10
Nodes (25): ContextEntry, ContextStore, Transient context store for inter-agent communication.  Agents write compact out, In-memory store for agent context references.      Each entry is a pointer, not, Add a context entry, return its ID., Look up a context entry by reference ID., Return all entries referencing a specific ticket., Return all entries from a specific agent. (+17 more)

### Community 52 - "Community 52"
Cohesion: 0.10
Nodes (23): Handoff, Compact structured handoff between agents., Tests for the Handoff protocol and compact serialization., Unknown / empty phase strings degrade to the default instead of raising., Building a Handoff from an invalid phase string no longer crashes., Flags are included in the serialized message., Handoff with ticket_ids serializes them in the compact key., Handoff with context_refs includes them in the compact format. (+15 more)

### Community 53 - "Community 53"
Cohesion: 0.13
Nodes (25): _build_serde(), get_checkpointer(), get_pool(), LangGraph checkpointer setup using AsyncPostgresSaver.  We expose a context-mana, Return a serializer that knows about our application state types.      ``allowed, Async context manager that yields an initialized AsyncPostgresSaver.      On fir, _agent_from_wrote_nodes(), _cache_bytes_total() (+17 more)

### Community 54 - "Community 54"
Cohesion: 0.10
Nodes (19): 1A. Trim AI messages on checkpoint write, 1B. Remove dead SystemState fields, 1C. Single canonical project goal, 2A. Compact handoff format, 2B. Smart context lookup via context_refs, 2C. Agent-to-agent direct handoff, 3A. Static system prompt cache, 3B. Skill loading optimization (+11 more)

### Community 55 - "Community 55"
Cohesion: 0.23
Nodes (10): close_pool(), create_app(), lifespan(), FastAPI application entrypoint.  Wires HTTP routes, the WebSocket hub, CORS, DB, WebSocket endpoint that streams every EventBus event to connected clients., ws_endpoint(), init_db(), Create all tables (used in dev / CI; production should use Alembic). (+2 more)

### Community 56 - "Community 56"
Cohesion: 0.06
Nodes (39): get_cached_lead_appendix(), get_cached_role_base(), Centralized system prompts for every agent role.  Kept in one place so they can, Return the RITE + tool contract appendix (cached once)., Return the static base for a given role (cached once).      Only called at app s, Return the RITE + tool contract appendix (cached once)., Return the static base for a given role (cached once).      Only called at app s, Return the RITE + tool contract appendix (cached once). (+31 more)

### Community 57 - "Community 57"
Cohesion: 0.12
Nodes (16): Agent workflow and context, Already in the design, Checkpointed state (`SystemState`), Efficiency and reducing context, External persistence, Highest-impact improvements, How context is carried between steps, Key source files (+8 more)

### Community 58 - "Community 58"
Cohesion: 0.12
Nodes (15): 1. Configure, 2. Boot the stack, 3. Use it, Agent hierarchy, API endpoints, Architecture, How TDD is enforced, Local development without Docker (+7 more)

### Community 59 - "Community 59"
Cohesion: 0.10
Nodes (14): PM step 5 must review subtask count per ticket., PM step 5 must review subtask count per ticket., Backend lead must have minimum subtask count constraint., Merged lead must have minimum subtask count constraint and decomposition guidanc, Frontend lead must have minimum subtask count constraint., Verify the PM and lead prompts force fine-grained decomposition., Verify the PM and lead prompts force fine-grained decomposition., PM must explicitly say NEVER bundle into one ticket. (+6 more)

### Community 60 - "Community 60"
Cohesion: 0.18
Nodes (9): HandoffV2, Phase, Structured inter-agent handoff protocol.  Replaces the natural-language handoff, Create a Handoff from a RoutingDecision., Extended handoff that carries a small ContextStore snapshot.      Instead of ful, Best-effort map an arbitrary phase string to a valid Phase.          The PM mode, Serialize to a compact message format for the agent., Enum (+1 more)

### Community 61 - "Community 61"
Cohesion: 0.13
Nodes (47): AgentRole, AnswerQuestion, SubtaskCreate, SubtaskStatus, SubtaskUpdate, TicketCreate, TicketUpdate, TodoCreate (+39 more)

### Community 62 - "Community 62"
Cohesion: 0.12
Nodes (12): Verify dev prompts include test failure diagnosis instructions., Verify dev prompts include test failure diagnosis instructions., Dev prompt must instruct using shell_run for diagnosis., Dev prompt must instruct using shell_run for diagnosis., Dev prompt must stop after 3 identical failures., Dev prompt must stop after 3 identical failures., Dev prompt must check environment before running tests., Dev prompt must check environment before running tests. (+4 more)

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
Cohesion: 0.39
Nodes (8): bool, Ticket, Conservative readiness check for moving IN_REVIEW -> TODO.      User expectation, Conservative readiness check for moving IN_REVIEW -> TODO.      User expectation, Conservative readiness check for moving IN_REVIEW -> TODO.      User expectation, _text_suggests_client_scope(), _ticket_has_unanswered_questions(), _ticket_ready_for_todo()

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
Cohesion: 0.18
Nodes (18): _build_specialist_input(), Assemble the minimal message list a specialist needs to act.      Strategy:, Assemble the minimal message list a specialist needs to act.      Strategy:, SystemState, str, SystemState, _make_state(), Tests for _build_specialist_input changes (canonical project_context). (+10 more)

### Community 75 - "Community 75"
Cohesion: 0.29
Nodes (4): DEFAULT_COLORS, HitlPanel(), InterruptItem, KIND_COLORS

### Community 76 - "Community 76"
Cohesion: 0.12
Nodes (20): Root state object shared across the orchestration graph., Root state object shared across the orchestration graph., SystemState, Any, int, Specialist focused-history assembly., test_skips_duplicate_goal_when_project_context_set(), Tests for SystemState changes (dead fields neutralized, context_store added). (+12 more)

### Community 77 - "Community 77"
Cohesion: 0.18
Nodes (16): AgentRole, int, str, SubtaskStatus, Unit tests for actionable subtask selection helpers., _sub(), test_first_actionable_subtask_ignores_done(), test_first_actionable_subtask_picks_blocked_before_pending() (+8 more)

### Community 78 - "Community 78"
Cohesion: 0.25
Nodes (8): HitlBadge(), NAV, InterruptPanel(), AgentLogEntry, TicketT, CrashState, PendingInterrupt, UIState

### Community 79 - "Community 79"
Cohesion: 0.06
Nodes (45): build_lead_subgraph(), Root LangGraph orchestration graph.  The Project Manager acts as the supervisor;, Build the merged Lead agent subgraph (handles both backend and frontend planning, build_specialist_subgraph(), Build and compile a tool-using specialist subgraph., Build and compile a tool-using specialist subgraph., Build and compile a tool-using specialist subgraph.      When ``verify_completio, Classify whether the agent resolved the subtask it picked up.      Returns ``(st (+37 more)

### Community 80 - "Community 80"
Cohesion: 0.29
Nodes (6): Agent execution environment, Agent notes (this workspace), Framework stacks (e.g. Next.js), Native binaries vs npm, What not to do, Where documentation lives

### Community 81 - "Community 81"
Cohesion: 0.20
Nodes (11): build_root_graph(), Compile the full multi-agent orchestration graph., Compile the full multi-agent orchestration graph., BaseCheckpointSaver, get_state(), list_interrupts(), Best-effort JSON-safe rendering for stream updates., Best-effort JSON-safe rendering for stream updates. (+3 more)

### Community 84 - "Community 84"
Cohesion: 0.23
Nodes (15): describe_llm_slug(), log_llm_invoke_exception_context(), log_llm_invoke_start(), log_rag_crag_llm_targets(), Structured logging around LLM HTTP calls — see where traffic is sent., Return (provider, model_name, human_readable_endpoint_hint)., Call from inside an ``except`` block — attaches stack trace via ``exception()``., Call from inside an ``except`` block — attaches stack trace via ``exception()``. (+7 more)

### Community 85 - "Community 85"
Cohesion: 0.33
Nodes (4): str, Application configuration loaded from environment via pydantic-settings., Settings, BaseSettings

### Community 86 - "Community 86"
Cohesion: 0.17
Nodes (20): _engaged_subtask_id_from_tools(), _latest_subtask_id_from_tools(), _parse_next_pending_tool_result(), Generic tool-using agent runner shared by specialist subgraphs.  Each specialist, Subtask the agent picked up via ``next_pending*`` this turn (``None`` = no work), IDs of subtasks the agent flipped to ``done`` via update_subtask_status., IDs of subtasks the agent flipped to ``done`` via update_subtask_status., Map subtask status → ids the agent set via ``update_subtask_status``. (+12 more)

### Community 87 - "Community 87"
Cohesion: 0.26
Nodes (12): Any, str, complete_assignment(), _dump(), Persistence tools for Coordinator agent.  These tools handle all database writes, Mark a subtask as done and transition related ticket if ready.      USE WHEN: De, Create or update a ticket in the database.      USE WHEN: Persisting a new ticke, Transition a ticket to a new status.      USE WHEN: Moving tickets through the w (+4 more)

### Community 88 - "Community 88"
Cohesion: 0.33
Nodes (8): AsyncSession, ProjectCreate, str, create(), delete_one(), get_one(), list_all(), Delete a project and every artifact attached to it.      Cascades:     - Cancels

### Community 89 - "Community 89"
Cohesion: 0.40
Nodes (4): str, ask_human(), Human-in-the-loop tool that pauses graph execution via ``interrupt()``.  Calling, Ask the human supervisor a question and wait for the answer.      Use sparingly

### Community 90 - "Community 90"
Cohesion: 0.25
Nodes (8): create_subtask(), A single RITE-format test case the lead must specify per subtask., A single RITE-format test case the lead must specify per subtask., Create an ordered subtask under a ticket using RITE-format test cases.      Idem, A single RITE-format test case the lead must specify per subtask., Create an ordered subtask under a ticket using RITE-format test cases.      USE, Create an ordered subtask under a ticket using RITE-format test cases.      USE, TestCaseInput

### Community 91 - "Community 91"
Cohesion: 0.31
Nodes (8): str, datetime, Base, Shared declarative base for all ORM models., DeclarativeBase, SQLAlchemy ORM models for the ticket platform.  Hierarchy: Project → Ticket → Su, _utcnow(), _uuid()

### Community 92 - "Community 92"
Cohesion: 0.25
Nodes (8): AgentRole, list_subtasks(), mark_todo_done(), Mark a todo as completed., Mark a todo as completed., Mark a todo as completed., List subtasks filtered by project_id and optionally ticket_id and/or assigned_to, List subtasks filtered by project_id and optionally ticket_id and/or assigned_to

### Community 94 - "Community 94"
Cohesion: 0.24
Nodes (12): int, MonkeyPatch, str, Exception, DummyRunnable, Provider400Error, Minimal Runnable-like object for retry wrapper tests., test_generic_400_does_not_sleep_or_retry() (+4 more)

### Community 95 - "Community 95"
Cohesion: 0.25
Nodes (8): Update a subtask's status. Valid: pending, in_progress, done, blocked., Patch fields on an existing subtask. Lead-only tool.      Use this when an exist, Update a subtask's status. Valid: pending, in_progress, done, blocked., Update a subtask's status. Valid: pending, in_progress, done, blocked., Patch fields on an existing subtask. Lead-only tool.      Use this when an exist, Patch fields on an existing subtask. Lead-only tool.      Use this when an exist, update_subtask(), update_subtask_status()

### Community 96 - "Community 96"
Cohesion: 0.40
Nodes (4): bool, Smoke tests for the ticket-system state-machine logic.  These exercise the pure-, test_transition_allowed(), TicketStatus

### Community 97 - "Community 97"
Cohesion: 0.24
Nodes (8): log_resolved_llm_routing(), Log where each configured role sends traffic (local vs cloud).      Misconfigura, Log where each configured role sends traffic (local vs cloud).      Misconfigura, Log where each configured role sends traffic (local vs cloud).      Misconfigura, Log where each configured role sends traffic (local vs cloud).      Misconfigura, _split_slug(), test_split_slug_explicit_openrouter(), test_split_slug_known_direct_providers()

### Community 98 - "Community 98"
Cohesion: 0.67
Nodes (6): _is_provider_400_error(), _should_retry_transient(), _status_code_from_exc(), _transient_exceptions(), BaseException, bool

### Community 99 - "Community 99"
Cohesion: 0.33
Nodes (5): add_todo_to_subtask(), LangChain tools that wrap the ticket-system service for use by agents.  These to, Append a todo (low-level step) to a subtask., Append a todo (low-level step) to a subtask., Append a todo (low-level step) to a subtask.

### Community 100 - "Community 100"
Cohesion: 0.40
Nodes (5): Conditional edge: dispatch from the project manager based on its decision., Conditional edge: dispatch from the project manager based on its decision., _route_from_pm(), str, SystemState

### Community 101 - "Community 101"
Cohesion: 0.50
Nodes (4): _last_text(), Best-effort textual summary of the final AIMessage in ``messages``., Best-effort textual summary of the final AIMessage in ``messages``., BaseMessage

### Community 102 - "Community 102"
Cohesion: 0.50
Nodes (4): add_question_to_ticket(), Append a clarification question to a ticket and mark it questions_pending., Append a clarification question to a ticket and mark it questions_pending., Append a clarification question to a ticket and mark it questions_pending.

### Community 103 - "Community 103"
Cohesion: 0.50
Nodes (4): create_ticket(), Create a new ticket in DRAFT status owned by the given project.      USE WHEN: Y, Create a new ticket in DRAFT status owned by the given project.      USE WHEN: Y, Create a new ticket in DRAFT status owned by the given project.

### Community 104 - "Community 104"
Cohesion: 0.50
Nodes (4): delete_subtask(), Delete a subtask that's wrong or no longer needed. Lead-only tool.      Refuses, Delete a subtask that's wrong or no longer needed. Lead-only tool.      Refuses, Delete a subtask that's wrong or no longer needed. Lead-only tool.      Refuses

### Community 105 - "Community 105"
Cohesion: 0.50
Nodes (4): Transition a ticket to a new status. Valid: draft, in_review, questions_pending,, Transition a ticket to a new status. Valid: draft, in_review, questions_pending,, Transition a ticket to a new status. Valid: draft, in_review, questions_pending,, update_ticket_status()

## Knowledge Gaps
- **177 isolated node(s):** `nextConfig`, `config`, `name`, `version`, `private` (+172 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_cached_role_base()` connect `Community 56` to `Community 73`, `Community 39`?**
  _High betweenness centrality (0.137) - this node is a cross-community bridge._
- **Why does `get_settings()` connect `LLM Routing & RAG Pipeline` to `Agent Graph & Specialist Subgraphs`, `Message Reducer & Common Utils`, `Researcher Subgraph & Code Tools`, `Skills Registry System`, `Agent API Routes & Checkpoint Cache`, `Checkpointer & App Lifecycle`, `Observability & Stream Helpers`, `Checkpoint List Cache`, `Community 19`, `Community 22`, `Community 27`, `Community 38`, `Community 53`, `Community 55`, `Community 79`, `Community 84`, `Community 85`, `Community 88`, `Community 97`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `SubtaskStatus` connect `Community 61` to `Event Bus & Ticket Service`, `Project Manager & Routing`, `Agent Logs & EventBus Persistence`, `Checkpoint List Cache`, `Community 27`, `Community 39`, `Community 68`, `Community 69`, `Community 74`, `Community 76`, `Community 77`, `Community 79`, `Community 86`, `Community 87`, `Community 90`, `Community 91`, `Community 92`, `Community 99`, `Community 101`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Are the 81 inferred relationships involving `SubtaskStatus` (e.g. with `bool` and `int`) actually correct?**
  _`SubtaskStatus` has 81 INFERRED edges - model-reasoned connections that need verification._
- **Are the 72 inferred relationships involving `AgentRole` (e.g. with `bool` and `int`) actually correct?**
  _`AgentRole` has 72 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `Event` (e.g. with `AgentEvent` and `AIMessage`) actually correct?**
  _`Event` has 46 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `TicketStatus` (e.g. with `bool` and `int`) actually correct?**
  _`TicketStatus` has 58 INFERRED edges - model-reasoned connections that need verification._