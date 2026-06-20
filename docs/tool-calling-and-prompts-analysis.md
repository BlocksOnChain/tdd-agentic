# Tool Calling and System Prompt Analysis

**Last updated:** 2026-06-20

## Executive Summary

The TDD Agentic system uses modern prompt engineering techniques (structured tool args via Pydantic, focused specialist input, condensed PM history, XML context separation, phased dev toolsets). **Nine high-priority items from this analysis are now implemented** (see [Implementation Status](#implementation-status) below).

Remaining gaps cluster in three areas:

1. **Semantic stop enforcement** — PM and dev agents rely on prompt text; step caps (`range(8)` / `max_steps`) still bound turns in code
2. **Prompt completeness** — few-shot examples, Coordinator/QA constraint blocks, and common-errors coverage on hitl/shell/researcher tools
3. **Context store adoption** — `context_refs` are wired in the handoff protocol but agents rarely populate them in practice

---

## Implementation Status

| # | Item | Status | Current code | Recommendation (if OPEN) |
|---|------|--------|--------------|----------------------------|
| 1 | PM structured routing | **DONE** | `with_structured_output(RoutingDecision)` on final turn via `_resolve_routing_decision()`; `_validate_ticket_ids()` strips hallucinated UUIDs | — |
| 2 | Lead → `execution_plan` | **DONE** | Lead subgraph `parse_execution_plan=True`; `_parse_execution_plan()` in `runner.py` writes `state.execution_plan` | — |
| 3 | RITE contract in `LEAD_SYSTEM` | **DONE** | `_RITE_CONTRACT` appended to `LEAD_SYSTEM`; `LEAD_PLANNING_APPENDIX` unchanged (`_RITE_CONTRACT + _LEAD_TOOL_CONTRACT`) | — |
| 4 | `get_ticket_summary` for PM | **DONE** | Added to `PM_TICKET_TOOLS`; docstring directs PM away from `get_ticket(detail='full')` for routing | — |
| 5 | `list_tickets` etag caching | **DONE** | `since_last_check` param + `_ticket_roster_etag()`; returns `{"unchanged": true, "etag": "..."}` when roster unchanged | — |
| 6 | XML context for specialists | **DONE** | `_build_specialist_system_prompt()` in `runner.py` wraps `<agent>`, `<context>`, `<instructions>`; PM uses same pattern in `_build_system_prompt()` | — |
| 7 | Phased dev toolsets | **DONE** | `phased_code_tools=True` on all dev subgraphs; ticket tools first, code tools bound after `next_pending*` engages a subtask | — |
| 8 | `code_tools.py` docstrings | **DONE** | USE WHEN / AVOID WHEN / RETURNS on all six code tools | Extend pattern to hitl/shell/researcher tools |
| 9 | Researcher structured summary | **DONE** | `structured_summary=True` on researcher subgraph; optional JSON parsed by `_parse_structured_summary()` with prose fallback | — |
| 10 | PM semantic stop | **OPEN** | Prompt has `=== STOP CONDITION ===`; code still loops `for step_i in range(8)` until no tool calls, then structured output | Enforce "decided → stop calling tools" in code, not just prompt |
| 11 | Coordinator / QA constraints | **OPEN** | PM, Lead, Dev, Researcher have `=== CONSTRAINTS ===`; `COORDINATOR_SYSTEM` and `QA_SYSTEM` lack dedicated constraint blocks | Add constraint sections to Coordinator and QA-only extras |
| 12 | Few-shot examples | **OPEN** | Routing JSON and `create_subtask` described in prose only | Add one-shot routing JSON + `create_subtask` call example |
| 13 | PM explicit reasoning (code) | **OPEN** | Reasoning steps in `PROJECT_MANAGER_SYSTEM` prompt text only | No code enforcement; prompt-only is acceptable but fragile |
| 14 | Common errors in all tools | **PARTIAL** | ticket_tools + code_tools enriched; hitl, shell, researcher tools bare | Extend USE WHEN / Common errors to remaining tools |
| 15 | Context store refs | **OPEN** | `RoutingDecision.context_refs` → `Handoff`; format documented in skills seed | Agents rarely write refs; specialists don't resolve them yet |

### Architecture corrections (2026-06-20)

| Topic | Current code | Prior doc assumption |
|-------|--------------|----------------------|
| Lead tools | `tools=[]` — cognitive-only; outputs `execution_plan` JSON | Lead had 8+ ticket tools |
| Coordinator persistence | `PERSISTENCE_TOOLS` + `rag_query`; reads `state.execution_plan` | Lead persisted via ticket tools |
| `LEAD_TICKET_TOOLS` | Defined in `ticket_tools.py` but **unused** by Lead subgraph | Lead bound ticket tools |
| Lead → Coordinator handoff | `state.execution_plan` set by `_parse_execution_plan()`; Coordinator calls `save_execution_plan()` | Ad-hoc JSON in messages only |
| PM stop condition | Still `range(8)` step cap per turn; structured output only on **final** turn (no tool calls) | Semantic stop implied in code |

---

## 1. Tool Calling Analysis

### 1A. Current tool definition quality

The codebase uses LangChain `@tool` with Pydantic argschemas. Tool schemas are auto-discovered by the LLM.

#### Gap 1.1: Tool descriptions are too short and miss **when to call** and **when NOT to call**

| Scope | Current code | Recommendation |
|-------|--------------|----------------|
| `ticket_tools.py` | **DONE** — USE WHEN / AVOID WHEN / RETURNS / Common errors on PM and dev ticket tools | — |
| `code_tools.py` | **DONE** — USE WHEN / AVOID WHEN / RETURNS on all six tools | — |
| `rag_tools.py` | **PARTIAL** — USE WHEN / AVOID WHEN present | Add Common errors |
| `persistence_tools.py` | **PARTIAL** — USE WHEN only | Add AVOID WHEN / Common errors |
| `hitl_tools.py` (`ask_human`) | **OPEN** — short description only | Add decision context |
| Researcher inline tools (`create_skill`) | **OPEN** — signature-level docstring | Add USE WHEN / RETURNS |

**Example (implemented in ticket_tools.py):**
```python
@tool
async def get_ticket(ticket_id: str, detail: str = "summary") -> str:
    """Fetch one ticket by UUID.

    USE WHEN: You need to read subtasks or RITE test cases.
    AVOID WHEN: You just need status/subtask_count (use list_tickets instead).
    AVOID WHEN: You are the PM making a routing decision — use list_tickets + get_ticket_summary instead.
    ...
    """
```

#### Gap 1.2: Tool return value format is not declared to the LLM

| Scope | Current code | Recommendation |
|-------|--------------|----------------|
| ticket + code tools | **DONE** — RETURNS sections in docstrings | — |
| persistence / rag / hitl / researcher | **OPEN** — no RETURNS sections | Add RETURNS / ON ERROR |

#### Gap 1.3: No tool selection guidance (decision tree)

| Agent | Current code | Recommendation |
|-------|--------------|----------------|
| PM | **DONE** — `=== TOOL SELECTION GUIDE ===` in `PROJECT_MANAGER_SYSTEM` | — |
| Lead | N/A — no tools (`tools=[]`) | — |
| Coordinator | **PARTIAL** — `=== TOOL USAGE ===` lists three tools | Add decision tree for error recovery |
| Dev / QA | **PARTIAL** — phased toolsets reduce confusion; TDD flow in `DEV_SYSTEM_BASE` | Add explicit "phase 1 → phase 2" guide in prompt |

### 1B. Tool calling patterns across agents

| Agent | Tool count | Tools used | Current code | Recommendation |
|-------|-----------|------------|--------------|----------------|
| PM | 7+2 | `PM_TICKET_TOOLS` (incl. `get_ticket_summary`) + `rag_query`, `ask_human` | Etag caching on `list_tickets`; structured routing on final turn | Semantic stop when decision reached (not just step cap) |
| Lead | 0 | None — cognitive-only | `parse_execution_plan=True`; `LEAD_TICKET_TOOLS` defined but unused | Remove or repurpose dead `LEAD_TICKET_TOOLS` export |
| Coordinator | 4+1 | `PERSISTENCE_TOOLS` + `rag_query` | Reads `state.execution_plan` from Lead | Add constraint block; narrow tool guidance |
| Dev | 6+6+1 (phased) | `DEV_TICKET_TOOLS` then `CODE_TOOLS` + `rag_query` | `phased_code_tools=True` — code tools bound after subtask engagement | — |
| Researcher | 7 | search, rag, fs, create_skill | Optional structured summary JSON | Enrich `create_skill` docstring |
| QA | 6+6+1 (phased) | Same phased pattern as dev | Same as dev subgraphs | QA-specific constraint section |

**Phased dev toolsets (implemented):**
```python
# Phase 1: ticket tools only (get next subtask)
DEV_TICKET_TOOLS = [next_pending_subtask_in_project, update_subtask_status, mark_todo_done, ...]

# Phase 2: code tools bound after next_pending* returns a subtask
code_tools=[*CODE_TOOLS, rag_query], phased_code_tools=True
```

### 1C. Tool error handling

| Scope | Current code | Recommendation |
|-------|--------------|----------------|
| ticket_tools (create/update) | **DONE** — "Common errors to avoid" in docstrings | — |
| code_tools | **DONE** — AVOID WHEN covers common mistakes | — |
| Runtime errors | Returned as JSON in ToolMessages; LLM retries | Anticipatory prevention in remaining tool docstrings |

---

## 2. System Prompt Analysis

### 2A. Current prompt structure

| Agent | Static base | Dynamic append | XML context | Total (~) |
|-------|-------------|----------------|-------------|-----------|
| PM | `PROJECT_MANAGER_SYSTEM` (~6KB) + skills + stack policy | `project_context`, active IDs | **DONE** — `_build_system_prompt()` | ~8–9KB |
| Lead | `LEAD_SYSTEM` (~4KB) + `_RITE_CONTRACT` + stack policy | skills, context via runner | **DONE** — `_build_specialist_system_prompt()` | ~7–8KB (down from ~9–10KB; no tool contract in active prompt) |
| Coordinator | `COORDINATOR_SYSTEM` (~1KB) | skills, context via runner | **DONE** — runner XML builder | ~2–3KB |
| Dev | `DEV_SYSTEM_BASE` + role + constraints | skills, context via runner | **DONE** — runner XML builder | ~4–5KB |
| Researcher | `RESEARCHER_SYSTEM` + constraints | skills, context via runner | **DONE** — runner XML builder | ~3–4KB |

**Note:** `LEAD_PLANNING_APPENDIX` (`_RITE_CONTRACT + _LEAD_TOOL_CONTRACT`) remains cached for legacy/tests but is **not** injected into the active Lead subgraph prompt. RITE format is inlined via `_RITE_CONTRACT` append on `LEAD_SYSTEM` only.

### 2B. Prompt engineering patterns currently used

- **Role anchoring** ("You are the X") ✓
- **Structured output protocol** (routing JSON, execution_plan JSON) ✓
- **Negative constraints** (forbidden subtask types) ✓
- **Resume safety instructions** ✓
- **Idempotency notes** ✓
- **Structured tool args via Pydantic** ✓
- **XML-style context separation** ✓ (PM + all specialists via runner)
- **PM structured routing via Pydantic** ✓ (`with_structured_output(RoutingDecision)`)
- **Phased tool binding for devs** ✓
- **PM tool selection guide** ✓
- **PM explicit reasoning steps (prompt text)** ✓
- **PM / Dev / Lead / Researcher constraint sections** ✓

### 2C. Modern prompt engineering patterns — status

#### Pattern 1: XML-style delimiters for context separation — **DONE**

| Location | Current code | Recommendation |
|----------|--------------|----------------|
| PM | `_build_system_prompt()` — `<agent>`, `<context>`, `<instructions>` | — |
| Specialists | `_build_specialist_system_prompt()` in `runner.py` | — |

#### Pattern 2: Explicit reasoning step before decisions — **PARTIAL**

| Agent | Current code | Recommendation |
|-------|--------------|----------------|
| PM | Reasoning steps 1–5 in `PROJECT_MANAGER_SYSTEM`; not enforced in code | Optional: validate tool-call sequence before accepting routing |
| Lead | Workflow steps in prompt; no tools to call | — |

#### Pattern 3: Few-shot examples for critical operations — **OPEN**

| Operation | Current code | Recommendation |
|-----------|--------------|----------------|
| PM routing JSON | Prose schema only | Add one complete routing JSON example |
| `create_subtask` | Described in `_LEAD_TOOL_CONTRACT` (unused appendix) | Add few-shot to Coordinator docs or persistence tool docstring |

#### Pattern 4: Explicit stop condition — **PARTIAL**

| Agent | Current code | Recommendation |
|-------|--------------|----------------|
| PM | Prompt: "Output routing JSON when you have decided"; code: `range(8)` cap + structured output on final no-tool-call turn | Enforce semantic stop in loop (break early when decision fields populated) |
| Dev | Prompt: ONE subtask per turn; code: `max_steps` + `require_subtask_resolution` gate | — |
| Lead | Prompt: JSON only, no tools; code: `parse_execution_plan=True`, `tools=[]` | — |

#### Pattern 5: Constraint section — **PARTIAL**

| Agent | Current code | Recommendation |
|-------|--------------|----------------|
| PM, Lead, Dev, Researcher | `=== CONSTRAINTS ===` at end of prompt | — |
| Coordinator | No dedicated constraint block | Add `=== CONSTRAINTS ===` (no planning, no UUID invention, etc.) |
| QA | Inherits dev constraints; no QA-only extras | Add integration/e2e scope constraints |

#### Pattern 6: Schema-first tool binding — **DONE (PM)**

| Agent | Current code | Recommendation |
|-------|--------------|----------------|
| PM | `with_structured_output(RoutingDecision)` on final turn; `_parse_routing()` kept as fallback | — |
| Lead | `_parse_execution_plan()` extracts JSON from final message | Consider `with_structured_output(ExecutionPlan)` on final turn |
| Researcher | `_parse_structured_summary()` with prose fallback | Optional: enforce via structured output |

### 2D. Prompt-specific improvements per agent

#### PM (highest impact — most calls, most expensive)

| Issue | Current code | Recommendation | Status |
|-------|--------------|----------------|--------|
| Routing output | `with_structured_output(RoutingDecision)` + `_validate_ticket_ids` | — | **DONE** |
| Context separation | XML-style in `_build_system_prompt()` | — | **DONE** |
| Decision reasoning | Prompt text only | Code validation optional | **PARTIAL** |
| Constraint visibility | `=== CONSTRAINTS ===` section | — | **DONE** |
| Stop condition | Prompt semantic + `range(8)` code cap | Enforce semantic stop in code | **OPEN** |
| Ticket ID validation | `_validate_ticket_ids()` strips hallucinated UUIDs | — | **DONE** |
| list_tickets caching | etag / `since_last_check` | — | **DONE** |

#### Lead (second highest — planning agent)

| Issue | Current code | Recommendation | Status |
|-------|--------------|----------------|--------|
| Tool binding | `tools=[]` — cognitive-only | — | **DONE** |
| RITE contract | Inlined in `LEAD_SYSTEM` via `_RITE_CONTRACT` | — | **DONE** |
| execution_plan parsing | `_parse_execution_plan()` → `state.execution_plan` | — | **DONE** |
| `_LEAD_TOOL_CONTRACT` | In `LEAD_PLANNING_APPENDIX` only; not in active prompt | Remove dead appendix or document as reference | **N/A** |
| Few-shot subtask creation | Description-based | Add example in `LEAD_SYSTEM` | **OPEN** |
| Cross-agent boundary | Verbal domain list in prompt | Explicit SCOPE sections | **OPEN** |

#### Coordinator (persistence agent)

| Issue | Current code | Recommendation | Status |
|-------|--------------|----------------|--------|
| Role separation | `PERSISTENCE_TOOLS` only; no planning | — | **DONE** |
| execution_plan input | Reads `state.execution_plan` from Lead | — | **DONE** |
| Constraint section | Missing | Add `=== CONSTRAINTS ===` | **OPEN** |

#### Dev (most frequent — called per subtask)

| Issue | Current code | Recommendation | Status |
|-------|--------------|----------------|--------|
| TDD cycle | Text in `DEV_SYSTEM_BASE` | Flow chart optional | **PARTIAL** |
| Tool selection | Phased: ticket → code after subtask engagement | — | **DONE** |
| Stop condition | ONE subtask + `require_subtask_resolution` gate | — | **DONE** |
| Verification gate | `verify_completion=True` reverts unverified done | — | **DONE** |

#### Researcher

| Issue | Current code | Recommendation | Status |
|-------|--------------|----------------|--------|
| Structured output | Optional JSON via `structured_summary=True` | — | **DONE** |
| Scope | Broad prompt | Decision tree optional | **OPEN** |
| Skill creation | Bare `create_skill` docstring | Enrich docstring | **OPEN** |

---

## 3. Critical Path: PM Routing Decision

The PM's routing decision is the single most important prompt in the system. Every turn depends on it.

### Current routing flow (2026-06-20):

1. PM gets XML-wrapped system prompt + condensed messages
2. PM may call ticket tools (`list_tickets` with etag, `get_ticket_summary`, etc.) — up to **8 rounds** (`range(8)`)
3. On turn with **no tool calls**, PM invokes `with_structured_output(RoutingDecision)` via `_resolve_routing_decision()`
4. Fallback: regex JSON extraction via `_parse_routing()` if structured output fails
5. `_validate_ticket_ids()` strips hallucinated UUIDs; falls back to UUIDs in `instructions`
6. `_format_pm_handoff()` serializes to compact Handoff protocol (supports `context_refs`)
7. Next agent reads the handoff message

### Lead → Coordinator flow (2026-06-20):

1. PM routes to Lead with ticket instructions
2. Lead (`tools=[]`) outputs `execution_plan` JSON only
3. `_parse_execution_plan()` writes `state.execution_plan`
4. PM routes to Coordinator
5. Coordinator calls `save_execution_plan()` via `PERSISTENCE_TOOLS`

### Failure modes:

| Failure | Root cause | Current code | Recommendation |
|---------|-----------|--------------|----------------|
| Model adds prose around JSON | No structured output | **FIXED** — `with_structured_output` on final turn | — |
| Model guesses UUIDs | No schema validation | **FIXED** — `_validate_ticket_ids()` | — |
| Model outputs routing alongside tool calls | Unclear stop | Prompt says mutually exclusive; code accepts last no-tool turn | Enforce in loop |
| Model ignores tool results | Instructions mixed with context | **FIXED** — XML separation | — |
| PM re-fetches unchanged tickets | No caching | **FIXED** — etag / `since_last_check` | — |
| Lead tool-call errors | Too many tools | **FIXED** — Lead is tool-free; Coordinator persists | — |
| Context refs unused | No agent writes refs | Handoff format supports `context_refs` | Wire ContextStore writes in specialists |

---

## 4. Recommendations Priority

### Critical — status

| # | Recommendation | Status | Current code |
|---|----------------|--------|--------------|
| 1 | PM structured output | **DONE** | `with_structured_output(RoutingDecision)` + fallback |
| 2 | PM context separation | **DONE** | XML in `_build_system_prompt()` |
| 3 | PM explicit stop condition | **OPEN** | Prompt only; code uses `range(8)` |

### High — status

| # | Recommendation | Status | Current code |
|---|----------------|--------|--------------|
| 4 | Tool description enrichment | **PARTIAL** | ticket + code done; hitl/shell/researcher bare |
| 5 | Constraint section per prompt | **PARTIAL** | PM/Lead/Dev/Researcher done; Coordinator/QA missing |
| 6 | Phased tool availability | **DONE** | `phased_code_tools=True` on all dev subgraphs |

### Medium — status

| # | Recommendation | Status | Current code |
|---|----------------|--------|--------------|
| 7 | Few-shot examples | **OPEN** | No routing JSON or create_subtask examples |
| 8 | Explicit reasoning step | **PARTIAL** | PM prompt text; not code-enforced |
| 9 | Anticipatory error prevention | **PARTIAL** | ticket + code tools; others bare |

### Low — status

| # | Recommendation | Status | Current code |
|---|----------------|--------|--------------|
| 10 | Structured research output | **DONE** | Optional JSON with prose fallback |
| 11 | Context store refs | **OPEN** | Handoff wired; agents don't write refs |
| 12 | Lead structured output | **OPEN** | JSON parse from prose; consider `with_structured_output` |

---

## 5. Cross-Cutting Modern Prompt Engineering Principles

### Principle 1: Separate context from instructions — **DONE**

PM and all specialists use XML-style delimiters via `_build_system_prompt()` / `_build_specialist_system_prompt()`.

### Principle 2: Schema-first for critical decisions — **DONE (PM)**

PM uses `with_structured_output(RoutingDecision)`. `_parse_routing()` retained as fallback only.

### Principle 3: Tool selection guidance > tool description — **PARTIAL**

PM has explicit tool selection guide. Dev phased toolsets reduce selection surface. Coordinator and researcher could use decision trees.

### Principle 4: Few-shot > one-shot > description — **OPEN**

No few-shot examples yet for routing JSON or subtask creation.

### Principle 5: Explicit stop > implicit limits — **PARTIAL**

Prompts define semantic stops; code still relies on step caps (`range(8)` for PM, `max_steps` for specialists). Dev agents have additional gates (`require_subtask_resolution`, `verify_completion`).

---

## Appendix: Key file references

| Concern | File | Symbol |
|---------|------|--------|
| PM structured routing | `backend/agents/project_manager/supervisor.py` | `RoutingDecision`, `_resolve_routing_decision`, `_validate_ticket_ids` |
| PM XML prompt | `backend/agents/project_manager/supervisor.py` | `_build_system_prompt` |
| Specialist XML prompt | `backend/agents/runner.py` | `_build_specialist_system_prompt` |
| Lead execution plan | `backend/agents/runner.py` | `_parse_execution_plan`, `parse_execution_plan=True` |
| Phased dev tools | `backend/agents/runner.py` | `phased_code_tools`, `_current_tool_list` |
| Lead subgraph | `backend/agents/leads/lead/subgraph.py` | `tools=[]` |
| Coordinator subgraph | `backend/agents/coordinator/subgraph.py` | `PERSISTENCE_TOOLS` |
| PM ticket tools | `backend/tools/ticket_tools.py` | `PM_TICKET_TOOLS`, `get_ticket_summary`, etag caching |
| Unused lead tools | `backend/tools/ticket_tools.py` | `LEAD_TICKET_TOOLS` (defined, not bound) |
| Researcher summary | `backend/agents/researcher/subgraph.py` | `structured_summary=True` |
| Handoff protocol | `backend/agents/handoff.py` | `Handoff`, `context_refs` |
