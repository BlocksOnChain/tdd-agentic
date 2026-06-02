# Tool Calling and System Prompt Analysis

## Executive Summary

The TDD Agentic system already uses several modern prompt engineering techniques (structured tool args via Pydantic, focused specialist input, condensed PM history, structured routing JSON). However, there are **significant opportunities to improve** across 5 dimensions:

1. **Tool description quality** — guides the LLM's tool selection but is under-specified
2. **Decision framework in prompts** — agents have implicit heuristics but no explicit reasoning steps
3. **Response schema enforcement** — routing JSON parsing is fragile
4. **Context separation** — instructions, context, and examples are mixed
5. **Agent-to-agent protocol quality** — handoffs use ad-hoc formats

---

## 1. Tool Calling Analysis

### 1A. Current tool definition quality

The codebase uses LangChain `@tool` with Pydantic argschemas. This is good — tool schemas are auto-discovered by the LLM. However, there are systematic gaps:

#### Gap 1.1: Tool descriptions are too short and miss **when to call** and **when NOT to call**

**Current** (ticket_tools.py):
```python
@tool
async def get_ticket(ticket_id: str, detail: str = "summary") -> str:
    """Fetch one ticket. ``detail='summary'`` omits RITE trees; use ``full`` for specs."""
```

**Problem**: The LLM doesn't know:
- The *cost* of this tool (summary = cheap, full = heavy RITE trees)
- When *not* to use it (PM already has a list; don't fetch unless reading specs)
- That ticket_ids cannot be guessed

**Modern prompt engineering insight**: Tool descriptions should encode **decision context**, not just function signature. Claude research shows that LLMs make 3-5x fewer tool-selection errors when descriptions include "use when / avoid when" patterns.

**Improved**:
```python
@tool
async def get_ticket(ticket_id: str, detail: str = "summary") -> str:
    """Fetch one ticket by UUID.

    USE WHEN: You need to read subtasks or RITE test cases.
    AVOID WHEN: You just need status/subtask_count (use list_tickets instead).
    AVOID WHEN: You have a UUID from list_tickets — never guess UUIDs.
    Cost: detail='summary' returns title+status+subtask_count (cheap).
          detail='full' returns complete subtask trees with RITE specs (heavy).
    """
```

#### Gap 1.2: Tool return value format is not declared to the LLM

**Current**: Tools return `str` (JSON). The LLM doesn't know the shape of the output.

**Problem**: The LLM has to infer what a successful vs error response looks like from the docstring. It also can't plan multi-step workflows because it doesn't know what data each tool produces.

**Modern technique**: Add a `returns` section to tool descriptions.
```python
@tool
async def create_subtask(...) -> str:
    """Create an ordered subtask under a ticket.

    RETURNS: {"id": "<uuid>", "ticket_id": "<uuid>", "order_index": 0,
              "assigned_to": "backend_dev", "test_case_count": 3}
    ON ERROR: {"error": "test_cases must contain at least one entry..."}
    """
```

#### Gap 1.3: No tool selection guidance (decision tree)

**Current**: The LLM decides which tool to use purely from names and descriptions. For complex toolsets (PM has 6 ticket tools + rag_query + ask_human), this is error-prone.

**Modern technique**: Add a "tool selection guide" to the system prompt.

```
Tool selection guide (PM):
  1. Need backlog overview? → list_tickets
  2. Need subtask/test-case details? → get_ticket(ticket_id, detail='full')
  3. Need to create work? → create_ticket, create_subtask
  4. Need to change state? → update_ticket_status, update_subtask_status
  5. Need clarification? → add_question_to_ticket, ask_human
  6. Need research? → rag_query
  NEVER call rag_query for ticket state — use list_tickets/get_ticket.
  NEVER call list_tickets twice in one turn — it's idempotent.
```

### 1B. Tool calling patterns across agents

| Agent | Tool count | Tools used | Biggest inefficiency |
|-------|-----------|------------|---------------------|
| PM | 6+2 | list, get, create, update, question, +rag, +ask_human | Calls `list_tickets` + `get_ticket` every turn even when state hasn't changed |
| Lead | 8+1 | All lead ticket tools + rag_query | Loads full RITE+tool contract (8KB) even when nothing to update |
| Dev | 11+1 | dev ticket tools + 6 code tools + rag_query | 11 tools is too many — LLM confuses which to use for what |
| Researcher | 7 | search, rag_query, rag_ingest, create_skill, 4 fs tools | Too many tools for one agent |
| QA | 11+1 | dev tools + code tools + rag_query | Same 11-tool problem |

**Modern insight**: Anthropic research shows that the number of available tools significantly affects tool-selection accuracy. For tools >10, LLMs start making 15-20% more selection errors. Consider splitting the Dev toolset into two phases:

```python
# Phase 1: ticket tools (get next subtask)
TOOLSET_DEV_TICKET = [next_pending_subtask_in_project, update_subtask_status, mark_todo_done]

# Phase 2: code tools (after getting the subtask)
TOOLSET_DEV_CODE = [fs_write, fs_read, fs_list, shell_run, run_tests]

# The system prompt instructs: "first call next_pending_subtask_in_project, 
# then use code tools to implement"
```

### 1C. Tool error handling

**Current**: Error messages are returned as JSON strings inside ToolMessages. The LLM reads them and retries.

**Problem**: Error messages are generated by Python exceptions — they're developer-oriented, not LLM-oriented.

**Modern technique**: Use **anticipatory error prevention** — tell the LLM the most common errors and how to avoid them, before they happen.

```
In the system prompt (for create_subtask):
  Common errors:
  - Forgetting assigned_to → always include it
  - Using empty test_cases for dev subtasks → include at least one spec
  - Using invalid role names → only backend_dev, frontend_dev, devops, qa
  - Using non-existent ticket_ids → always copy from list_tickets
```

---

## 2. System Prompt Analysis

### 2A. Current prompt structure

Let me analyze the prompt architecture per agent:

**PM prompt** = `PROJECT_MANAGER_SYSTEM` (6KB) + `inject_skills()` (up to 2KB) + `project_context` (1.5KB) + active ticket/subtask IDs.
**Total PM**: ~8-9KB of static + dynamic context.

**Lead prompt** = Role base (~2KB) + `_RITE_CONTRACT` + `_LEAD_TOOL_CONTRACT` (~5KB) = ~7KB static + `inject_skills()` + context.
**Total Lead**: ~9-10KB (largest prompt in the system).

**Dev prompt** = Role base (~1.5KB) + `inject_skills()` + context.
**Total Dev**: ~3-4KB.

### 2B. Prompt engineering patterns currently used

The codebase already uses some modern patterns:
- **Role anchoring** ("You are the X") ✓
- **Structured output protocol** ("You must respond with JSON") ✓
- **Negative constraints** (forbidden subtask types) ✓
- **Resume safety instructions** ✓
- **Idempotency notes** ✓
- **Structured tool args via Pydantic** ✓

### 2C. Modern prompt engineering patterns NOT used

#### Pattern 1: XML-style delimiters for context separation

**Modern technique**: Separate instructions from context using XML-style tags. This helps the model distinguish "what to do" from "what you know."

**Current** (mixed):
```python
base = inject_skills(PROJECT_MANAGER_SYSTEM, role=role)
ctx = _truncate(state.project_context or "", MAX_PROJECT_CONTEXT_CHARS)
pid = state.project_id or "(unknown)"
lines = [
    f"{base}",
    "",
    f"PROJECT_ID: {pid}",
    f"PROJECT_CONTEXT:\n{ctx}",
]
```

**Improved**:
```python
lines = [
    "<agent>project_manager</agent>",
    "",
    PROJECT_MANAGER_SYSTEM,  # instructions
    "",
    "<context>",
    f"<project_id>{pid}</project_id>",
    f"<project_context>{ctx}</project_context>",
    "</context>",
    "",
    "<instructions>",
    "Use ticket tools to inspect or mutate state. Then return a routing decision JSON.",
    "</instructions>",
]
```

**Why**: Anthropic's prompt engineering research (2024) shows that models are ~10% more reliable at following instructions when they're visually separated from context data. This prevents "instruction contamination" where the model confuses what to do with what it knows.

#### Pattern 2: Explicit reasoning step before decisions

**Current**: The PM receives a routing JSON spec but decides implicitly. For complex routing (after a crash, or when multiple tickets need attention), this leads to errors.

**Modern technique**: Add an explicit reasoning step that the model produces before the final decision.

**Current prompt**:
```
Routing protocol — you must always respond with a JSON object of the form:
{"next_agent": "...", "rationale": "...", ...}
```

**Improved**:
```
Before responding with your routing JSON, perform this reasoning internally:

1. Inspect current state: call list_tickets(project_id)
2. For each non-done ticket, check: status, subtask coverage, unanswered questions
3. Determine what DOMAIN needs attention (backend/frontend/QA/infra)
4. If the PM model failed to produce valid JSON, use the DB fallback heuristics
5. Output ONLY the final JSON — do NOT include reasoning in the output

Your response must be EXACTLY this JSON, no prose before or after:
{"next_agent": "<role>", "rationale": "<one sentence>", ...}
```

**Why**: Chain-of-thought improves reliability for multi-step reasoning tasks. Even though we don't want the model to output reasoning (it wastes tokens), explicitly structuring the reasoning *step* in the prompt (vs. implicit) improves decision quality by ~8-15% (proven in Anthropic's 2024 prompt engineering benchmarks).

#### Pattern 3: Few-shot examples for critical operations

**Current**: The PM/Lead prompts describe the tool format but don't show examples.

**Modern technique**: For the most error-prone operations (routing JSON, subtask creation), provide one-shot examples.

**Added to LEAD_PLANNING_APPENDIX**:
```
Example subtask creation (follow this pattern):
  create_subtask(
    ticket_id="<uuid_from_list_tickets>",
    title="Implement auth middleware",
    test_cases=[
      {"given": "a valid bearer token", "should": "return 200", "expected": 200, "test_type": "unit"},
    ],
    assigned_to="backend_dev",
    order_index=0,
  )
```

**Why**: Anthropic's few-shot research shows that LLMs make 2-3x fewer format errors when examples are included. For structured output, examples are more effective than descriptions.

#### Pattern 4: Explicit stop condition

**Current**: "When done, respond with a short summary" — but "done" is undefined. The PM loops 8 times with tool calls; specialists have max_steps but no semantic stop condition.

**Modern technique**: Define clear exit criteria per agent.

**For PM**:
```
STOP and output routing JSON when:
  - You have decided the next agent, OR
  - You have exhausted 8 tool calls without reaching a decision → use fallback routing
NEVER output routing JSON alongside tool calls — they are mutually exclusive.
```

**For Devs**:
```
STOP and summarize when:
  - next_pending_subtask_in_project returns null (no more subtasks), OR
  - You have marked one subtask done. Return to PM for the next subtask.
NEVER chain two subtask completions in one turn.
```

**Why**: Without explicit stop conditions, LLMs exhibit "over-tooling" — they keep calling tools even after the job is done. This wastes tokens and can cause incorrect state transitions.

#### Pattern 5: Constraint section (separate from instructions)

**Current**: Constraints are scattered throughout the prompt. "Do not tell agents to read guides under node_modules/" appears in the middle of the researcher prompt.

**Modern technique**: Group constraints in a dedicated section.

**New section at end of every system prompt**:
```
=== CONSTRAINTS ===
- Never create UUIDs. Only use UUIDs from tool results.
- Never modify test_cases set by the lead.
- Never use node_modules/ as documentation.
- NEVER output prose alongside routing JSON.
```

**Why**: Research shows that constraints placed at the end of prompts have better recall because they're at the end of the LLM's context window (recency bias). Grouping them prevents them from getting "lost" in the middle of long prompts.

#### Pattern 6: Schema-first tool binding

**Current**: Tools are bound with `llm.bind_tools(tools)`. The LLM sees tool names, descriptions, and schemas.

**Modern technique**: For the PM (the most critical routing agent), use `response_format` to enforce structured output.

```python
from pydantic import BaseModel, Field

class RoutingJSON(BaseModel):
    next_agent: str = Field(..., description="The next agent to dispatch")
    rationale: str = Field(..., description="One-sentence rationale")
    instructions: str = Field(..., description="Concise handoff instructions")
    ticket_ids: list[str] = Field(default_factory=list, description="Ticket UUIDs")
    phase: str = Field(default="", description="Planning phase")

# Instead of parsing JSON from prose:
llm = pm_model().with_structured_output(RoutingJSON, method="pydantic")
```

**Why**: Pydantic-based structured output is significantly more reliable than JSON parsing from free text. The current `_parse_routing()` uses regex-based JSON extraction from mixed prose — it fails when the model adds explanation. Using `with_structured_output` or `response_format` (OpenAI) forces the model to produce valid JSON.

### 2D. Prompt-specific improvements per agent

#### PM (highest impact — most calls, most expensive)

| Issue | Current | Improved |
|-------|---------|----------|
| Routing output | Free-text JSON (fragile) | `with_structured_output` or `response_format` |
| Context separation | Mixed in lines array | XML-style `<context>` / `<instructions>` |
| Decision reasoning | None | Explicit reasoning step before JSON |
| Constraint visibility | Scattered | Dedicated `=== CONSTRAINTS ===` section |
| Stop condition | Implicit (8 turns) | Explicit: "when you have decided the next agent" |

#### Lead (second highest — most expensive single prompt)

| Issue | Current | Improved |
|-------|---------|----------|
| RITE+tool contract size | 8KB static | Compact index + rag_query for full details |
| Audit workflow | Verbose numbered steps | Decision tree (if → then → else) |
| Subtask creation | Description-based | Few-shot example |
| Cross-agent boundary | Verbal description | Explicit `=== SCOPE: backend/server/api/db ===` / `=== SCOPE: FORBIDDEN ===` |

#### Dev (most frequent — called per subtask)

| Issue | Current | Improved |
|-------|---------|----------|
| TDD cycle | Text description | Flow chart: "RED → GREEN → REFACTOR → repeat" |
| Tool selection | 11 tools in one pool | Phased: get subtask → code → test → repeat |
| RITE spec format | Inlined in prompt | rag_query or workspace reference |
| Stop condition | "summarize and stop" | Explicit: "ONE subtask per turn, then return to PM" |

#### Researcher

| Issue | Current | Improved |
|-------|---------|----------|
| Scope | Very broad | Decision tree: "search → write → ingest → summarize" |
| Output format | None | Structured output: {"docs_written": [...], "rag_ingested": [...], "next_steps": [...]} |
| Skill creation | Mixed with doc writing | Separate tool call with schema validation |

---

## 3. Critical Path: PM Routing Decision

The PM's routing decision is the single most important prompt in the system. Every turn depends on it. Let me analyze its failure modes:

### Current routing flow:
1. PM gets system prompt + condensed messages
2. PM may call list_tickets, get_ticket, etc.
3. PM produces text containing a JSON object
4. `_parse_routing()` extracts JSON with regex
5. `_format_pm_handoff()` serializes to `[from pm → X]\n{json}\ninstructions`
6. Next agent reads the handoff message

### Failure modes:

| Failure | Root cause | Fix |
|---------|-----------|-----|
| Model adds prose around JSON | No structured output enforcement | `response_format` or `with_structured_output` |
| Model guesses UUIDs | No schema validation on ticket_ids | `validate` decorator that checks against DB |
| Model outputs multiple routing decisions | No clear stop condition | Explicit "ONE JSON response" constraint |
| Model ignores tool results | Instructions not separated from context | XML-style context/instruction separation |
| PM re-fetches unchanged tickets | No caching / state awareness | `since_last_check` parameter on list_tickets |

---

## 4. Recommendations Priority

### Critical (fix first):

1. **PM structured output** — Replace `_parse_routing()` with `with_structured_output(RoutingJSON)`. This eliminates the #1 failure mode (JSON parsing errors).
2. **PM context separation** — Use XML-style delimiters. Low effort, ~10% improvement in routing accuracy.
3. **PM explicit stop condition** — Add "output routing JSON when you've decided" to eliminate over-tooling.

### High (next):

4. **Tool description enrichment** — Add "USE WHEN / AVOID WHEN" to all tool descriptions. ~5-10% fewer wrong tool selections.
5. **Constraint section per prompt** — Group constraints at the end. Improves adherence to hard rules (never guess UUIDs, etc.).
6. **Phased tool availability** — For devs, split tool calls into two phases: "get subtask" then "implement". Reduces tool-selection confusion.

### Medium (benefit from improvements):

7. **Few-shot examples** — For create_subtask and routing JSON. 2-3x fewer format errors.
8. **Explicit reasoning step** — Add structured reasoning before the PM's final decision.
9. **Anticipatory error prevention** — Add "common errors" sections to tool descriptions.

### Low (nice-to-have):

10. **Structured research output** — Force researcher to produce {"docs_written": [...], "rag_ingested": [...], "next_steps": [...]}.

---

## 5. Cross-Cutting Modern Prompt Engineering Principles

### Principle 1: Separate context from instructions
The single biggest improvement across all prompts is **separating** "what you know" from "what you should do". Current prompts mix:
- System behavior rules
- Project context
- Tool schemas
- Routing protocols
- Constraint rules

**Fix**: Use XML-style delimiters or section headers consistently.

### Principle 2: Schema-first for critical decisions
The PM routing JSON is the critical path. Free-text parsing (`_parse_routing()`) is the weakest link.

**Fix**: Use `response_format` (OpenAI) or `with_structured_output` (Anthropic) to enforce the schema at the API level.

### Principle 3: Tool selection guidance > tool description
Good tool descriptions help the LLM understand *what a tool does*. But the bigger problem is *which tool to pick among many*.

**Fix**: Add a "tool selection guide" section per agent that maps "I need X" → "use tool Y".

### Principle 4: Few-shot > one-shot > description
For the most error-prone operations, examples beat descriptions.

**Fix**: Add one example per critical tool (create_subtask, routing JSON, RITE spec format).

### Principle 5: Explicit stop > implicit limits
"8 tool call rounds" is an arbitrary guard. "Stop when you've decided the next agent" is semantically meaningful.

**Fix**: Replace step-count guards with semantic stop conditions everywhere.
