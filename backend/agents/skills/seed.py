"""Idempotent skill seeding executed at app startup.

Ensures the universal ``tdd-rite`` skill (and any other built-ins we add
later) is registered for every relevant agent role on every boot, even
when a fresh workspace is mounted.
"""
from __future__ import annotations

from backend.agents.skills.registry import get_skill_content, upsert_skill

TDD_RITE_NAME = "tdd-rite"
TDD_RITE_DESCRIPTION = (
    "Test-Driven Development discipline using the RITE assertion pattern "
    "(given/should/actual/expected). Defines the red-green-refactor cycle, "
    "test-type selection, and anti-patterns (no mocks in unit tests, no "
    "snapshots for logic verification)."
)
TDD_RITE_ROLES = [
    "lead",
    "backend_dev",
    "frontend_dev",
    "devops",
    "qa",
]

TDD_RITE_CONTENT = """# TDD with RITE — Working Discipline for AI Agents

You are practicing strict Test-Driven Development. Every change is driven
by ONE failing test at a time. The RITE assertion shape is the contract
between leads (who specify) and devs (who implement).

## The 5 questions every test must answer

1. What is the unit/component under test? (`describe(<unit>, ...)`)
2. What is the natural-language functional requirement? (`given` + `should`)
3. What is the actual output? (the unit was exercised by the test)
4. What is the explicit expected output? (`expected` is hard-coded)
5. How can we reproduce a failure? (answered implicitly when 1–4 are clear)

## RITE assertion shape

```js
assert({
  given: 'a subtotal and a coupon percent discount',
  should: 'return the discounted subtotal',
  actual: priceCalculator({ subtotal: 2000, couponPercent: 15 }),
  expected: 1700,
});
```

- `given` and `should` are natural language. Do NOT bake literal numbers
  into them — keep them readable as acceptance criteria.
- `expected` is the EXPLICIT literal value (or error type, or shape).
  Never use snapshot matchers for logic verification.

The Python equivalent (pytest) is a plain assertion plus a one-line
docstring describing the requirement:

```python
def test_price_calculator_applies_coupon_percent():
    \"\"\"given a subtotal and coupon percent discount, returns discounted subtotal.\"\"\"
    actual = price_calculator(subtotal=2000, coupon_percent=15)
    expected = 1700
    assert actual == expected, f"expected {expected}, got {actual}"
```

## Red → Green → Refactor (one requirement at a time)

1. Pick the NEXT unimplemented test case from `subtask.test_cases`.
2. Write exactly that test in the appropriate test file.
3. Run the test runner. The test MUST fail (red). If it passes
   immediately, your test is wrong (no determinism, missing assertion,
   etc.) — fix the test before writing code.
4. Implement the MINIMUM code to make the test pass. Resist adding
   anything not demanded by a failing test.
5. Run the test runner. All tests must pass (green).
6. Refactor for clarity. Keep the suite green.
7. Repeat for the next test case.

Never write multiple tests in a row before any of them passes. Never
implement code without a failing test first.

## Choose the right test type

| Type | When | Speed |
|------|------|-------|
| **unit** | Pure-function business logic, deterministic transforms, single component | < 10 ms each |
| **integration** | Two real collaborators wired together (service + repo, hook + store) | 30 ms – 30 s |
| **functional** | End-to-end user flow through public interfaces (HTTP, UI) | seconds – minutes |

Default to **unit**. Use integration only when the unit's whole point IS
collaboration. Functional tests are best left to QA + CI; agents that
spend turns waiting on full e2e flows time out.

## Make non-deterministic logic testable

Inject deterministic dependencies via optional parameters; fall back to
real ones when no override is passed:

```js
const createUser = ({
  name = 'Anonymous',
  email = '',
  timestamp = Date.now(),
  id = cuid(),
} = {}) => ({ type: 'createUser', payload: { name, email, timestamp, id } });
```

In tests, pass `timestamp` and `id` explicitly. In production, the
defaults run.

For Python, accept `now: Callable[[], datetime] = datetime.utcnow` and
`new_id: Callable[[], str] = lambda: str(uuid.uuid4())` as keyword args.

## Anti-patterns (do not do these)

- **Mocks inside unit tests.** A mock in a unit test is a smell: the unit
  is too tightly coupled to a side effect. Push the side effect to the
  edge (action dispatch, dependency injection) and unit-test the pure core.
- **Snapshot tests for logic verification.** Snapshots can guard visual
  output, never logic. Hard-code `expected`.
- **Multiple behaviours per test.** One test, one requirement.
- **Shared mutable fixtures.** Use factory functions per test instead.
- **Implementation in advance.** No production code without a failing test.
- **Skipping the red step.** If you didn't see it fail, you don't know it
  works.

## How leads should specify tests

Each `subtask.test_cases` entry MUST include:

- `given` — natural-language precondition (no literals)
- `should` — natural-language expected behaviour (no literals)
- `expected` — the EXPLICIT literal/error/shape the dev hard-codes
- `test_type` — one of `unit | integration | functional`
- `notes` — optional edge-case / fixture hints

Order the list so each test introduces ONE new requirement after the
previous one passes. Smallest first.

## How devs should consume tests

For each subtask:

1. Call `next_pending_subtask` and read `test_cases`.
2. For each entry IN ORDER:
   a. Write the test using the RITE shape, putting `expected` exactly as
      the lead specified.
   b. Run tests. Confirm RED (this specific test fails).
   c. Implement just enough code.
   d. Run tests. Confirm GREEN.
3. Mark the subtask DONE only after EVERY entry is green.

Never modify or delete the lead's test specs — if a spec looks wrong,
add a question to the ticket via the PM instead of silently editing.
"""


async def seed_builtin_skills() -> None:
    """Seed/refresh built-in skills. Safe to call on every startup."""
    if get_skill_content(TDD_RITE_NAME) != TDD_RITE_CONTENT:
        upsert_skill(
            name=TDD_RITE_NAME,
            description=TDD_RITE_DESCRIPTION,
            content=TDD_RITE_CONTENT,
            roles=TDD_RITE_ROLES,
            project_id=None,
        )


# ================================================================
# compact-output — teach agents to produce structured, compact summaries
# ================================================================

COMPACT_OUTPUT_NAME = "compact-output"
COMPACT_OUTPUT_DESCRIPTION = (
    "Produce compact, structured handoff summaries that other agents can consume "
    "efficiently. Defines the output format, context_refs pattern, and size limits."
)
COMPACT_OUTPUT_ROLES = [
    "project_manager",
    "researcher",
    "lead",
    "coordinator",
    "backend_dev",
    "frontend_dev",
    "devops",
    "qa",
]

COMPACT_OUTPUT_CONTENT = """# Compact Output — Structured Handoff Format

When producing summaries for other agents, use this compact format to
minimise token cost and maximise downstream agent efficiency.

## Handoff Summary Format

```
[from <agent_name> → <target_agent>]
{"t":"<target>","p":"<phase>","tik":["<uuid>"],"c":["ctx_<n>"]}
<one-line intent>
```

## Rules

1. **One line per handoff** — the intent (what to do) should fit in one line.
2. **Use context_refs** — instead of pasting full summaries, reference them:
   `ctx_<n>` is a pointer into the shared ContextStore.
3. **Keep handoffs under 200 chars** — if the intent is longer, split into
   multiple subtasks or write to workspace/rag.
4. **No re-stating the obvious** — don't repeat ticket bodies that are
   already in the DB. Reference the ticket UUID and let the agent look it up.

## Context Store Pattern

The PM maintains a shared ContextStore (in-memory). Agents write entries;
other agents reference by ID:

    ctx_id = context_store.add("researcher", "research_findings", ("uuid1",),
                               "Auth uses JWT with python-jose, 15min expiry")
    → returns "ctx_1"

Other agents receive {"context_refs": ("ctx_1",)} and can look up the
summary via context_store.lookup("ctx_1").

## Summary length

- Handoff intent: max 200 chars
- ContextStore entry: max 500 chars
- Full summaries can go to workspace/rag; keep the pointer in handoff

## Examples

Good:
    [from project_manager → backend_dev]
    {"t":"backend_dev","p":"implement","tik":["uuid1"],"c":["ctx_3"]}
    Implement JWT auth per ctx_3 (research) and ctx_7 (tech spec).

Bad:
    [from project_manager → backend_dev]
    {"ticket_ids":["uuid1"]}
    Implement JWT auth. The auth system should use JWT tokens with 15min expiry,
    refresh tokens with 7day expiry, stored in HTTP-only cookies. See research
    findings about using python-jose for JWT handling. The backend should also
    include middleware, endpoints, and validation...

The bad example is 4x the token cost of the good one for the same information.
"""

# ================================================================
# state-before-action — always check what's already done
# ================================================================

STATE_BEFORE_ACTION_NAME = "state-before-action"
STATE_BEFORE_ACTION_DESCRIPTION = (
    "Before starting any work, check current DB state to avoid duplicating effort. "
    "Defines the inspection pattern for tickets, subtasks, and workspace state."
)
STATE_BEFORE_ACTION_ROLES = [
    "project_manager",
    "lead",
    "coordinator",
    "backend_dev",
    "frontend_dev",
    "devops",
    "qa",
]

STATE_BEFORE_ACTION_CONTENT = """# State Before Action — Avoid Duplication

Before acting, ALWAYS check current state. The DB is the ground truth —
never rely on what you "remember" from earlier turns.

## Inspection Checklist

### Before creating tickets
1. Call `list_tickets(project_id)` — if a ticket with the same title exists,
   use the existing one (create_ticket is idempotent on project_id + title).

### Before creating subtasks
1. Call `get_ticket(ticket_id, detail='summary')` — check existing subtasks.
2. `create_subtask` is idempotent on (ticket_id, order_index) — each ticket
   may have only one subtask per order_index.

### Before starting implementation
1. Call `next_pending_subtask_in_project(project_id, role)` — get the next item.
2. Check `workspace/docs/` for existing architecture docs.
3. Use `rag_query` for library/pattern context.

### Before writing a summary
1. Check if a prior summary for this subtask exists in agent_logs.
2. If the DB already shows the subtask as "done", no need to summarise again.

### Before ending your turn (dev agents — mandatory)
1. Did `next_pending_subtask_in_project` return a subtask this turn?
   - **No** (null subtask): OK to summarize and stop — no work assigned.
   - **Yes**: you MUST either:
     a. `update_subtask_status(subtask_id, 'done')` after ALL specs pass and
        `run_tests` exited 0, OR
     b. `update_subtask_status(subtask_id, 'blocked')` with the exact blocker
        (missing toolchain, bad spec, unrecoverable test failure).
2. Never return "turn complete" while the subtask is still `in_progress`.
   An automated gate rewrites such handoffs as `[INCOMPLETE]` and PM re-dispatches you.
3. Prefer `run_tests` over exploratory `fs_read` once you know the target file paths.
   Environment setup (`node --version`, `npm install`) should happen once per project,
   not every turn — check AGENT EXECUTION ENVIRONMENT in your system prompt first.

## Pattern

    // 1. Check state
    list_tickets(project_id)
    get_ticket(ticket_id)

    // 2. Act only if needed
    create_subtask(...)  // only if state shows nothing exists

    // 3. Reference, don't repeat
    // Instead of pasting ticket body, reference by UUID
    "Implement per ticket <uuid>"

## Cost of not checking

- Each duplicate ticket: ~500 tokens wasted in tool output
- Each redundant summary: ~2000 tokens wasted in handoff
- Each re-created subtask: ~1000 tokens wasted + DB inconsistency

Always check before acting. It takes one tool call but saves hundreds of tokens.
"""

# ================================================================
# efficient-rag — smart context retrieval patterns
# ================================================================

EFFICIENT_RAG_NAME = "efficient-rag"
EFFICIENT_RAG_DESCRIPTION = (
    "Use RAG efficiently: query patterns, when to read from workspace vs vector store, "
    "and how to write high-quality docs that others can retrieve."
)
EFFICIENT_RAG_ROLES = [
    "project_manager",
    "researcher",
    "lead",
    "coordinator",
    "backend_dev",
    "frontend_dev",
    "devops",
    "qa",
]

EFFICIENT_RAG_CONTENT = """# Efficient RAG Usage — Query and Ingest Patterns

Use RAG for cross-agent knowledge transfer. Follow these patterns to
maximise retrieval quality and minimise token cost.

## Query Patterns

### Best queries (high recall)
- "JWT auth middleware implementation pattern" → finds auth docs
- "Next.js data fetching patterns" → finds architecture docs
- "database migration naming convention" → finds conventions docs

### Poor queries (low recall)
- "info" → too vague, returns noise
- "everything about auth" → too broad, wastes context budget
- "how to do auth" → generic, may miss project-specific docs

## When to Query vs Read

| Use case | Tool | Reason |
|----------|------|--------|
| Library/pattern search | `rag_query` | Vector search finds relevant chunks |
| Known file path | `fs_read` | Direct, no embedding cost |
| Project conventions | `rag_query` | May span multiple docs |
| Specific API contract | `rag_query` | Content indexed by vector similarity |

## Writing for Retrieval

When writing docs that others will need:

1. **Use descriptive titles** — "JWT + Refresh Token Auth Flow" not "Auth"
2. **Include key terms in the first 100 chars** — vector similarity weights early content
3. **Use consistent naming** — "Next.js" not "next" or "NextJS"
4. **Link related content** — reference other docs by path in prose
5. **Ingest after writing** — call `rag_ingest_text` immediately

## Chunk Quality

- Minimum ~200 chars per chunk for meaningful retrieval
- Avoid chunking single-line docs
- Use source path as metadata (appears in retrieval results)

## Cost Awareness

Each `rag_query` call:
- Embeds the query (vector operation)
- Runs relevance grader (small LLM call)
- May rewrite and re-query (additional LLM call)

Total: ~1-3 LLM calls per query. Use sparingly. Prefer `fs_read` when you
know the file path.
"""


async def seed_builtin_skills() -> None:
    """Seed/refresh built-in skills. Safe to call on every startup."""
    if get_skill_content(TDD_RITE_NAME) != TDD_RITE_CONTENT:
        upsert_skill(
            name=TDD_RITE_NAME,
            description=TDD_RITE_DESCRIPTION,
            content=TDD_RITE_CONTENT,
            roles=TDD_RITE_ROLES,
            project_id=None,
        )

    # ================================================================
    # caveman — ultra-compressed communication mode (always-on by default)
    # ================================================================

    CAVEMAN_NAME = "caveman"
    CAVEMAN_DESCRIPTION = (
        "Ultra-compressed communication mode. Cuts token usage ~75% by dropping filler, "
        "articles, and pleasantries while keeping full technical accuracy."
    )
    CAVEMAN_ROLES = [
        "project_manager",
        "researcher",
        "lead",
        "coordinator",
        "backend_dev",
        "frontend_dev",
        "devops",
        "qa",
    ]
    CAVEMAN_CONTENT = """# caveman

Respond terse like smart caveman. All technical substance stay. Only fluff die.

## Persistence

ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift.

## Rules

Drop: articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries,
hedging. Fragments OK. Short synonyms. Abbrev common terms (DB/auth/config/req/res/fn/impl).
Use arrows for causality (X -> Y). One word when one word enough.

Technical terms stay exact. Code blocks unchanged. Errors quoted exact.

Pattern: `[thing] [action] [reason]. [next step].`

## Auto-Clarity Exception

Drop caveman temporarily for: security warnings, irreversible action confirmations, multi-step
sequences where fragment order risks misread. Resume caveman after clear part done.
"""

    if get_skill_content(CAVEMAN_NAME) != CAVEMAN_CONTENT:
        upsert_skill(
            name=CAVEMAN_NAME,
            description=CAVEMAN_DESCRIPTION,
            content=CAVEMAN_CONTENT,
            roles=CAVEMAN_ROLES,
            project_id=None,
        )

    # New efficiency skills
    if get_skill_content(COMPACT_OUTPUT_NAME) != COMPACT_OUTPUT_CONTENT:
        upsert_skill(
            name=COMPACT_OUTPUT_NAME,
            description=COMPACT_OUTPUT_DESCRIPTION,
            content=COMPACT_OUTPUT_CONTENT,
            roles=COMPACT_OUTPUT_ROLES,
            project_id=None,
        )

    if get_skill_content(STATE_BEFORE_ACTION_NAME) != STATE_BEFORE_ACTION_CONTENT:
        upsert_skill(
            name=STATE_BEFORE_ACTION_NAME,
            description=STATE_BEFORE_ACTION_DESCRIPTION,
            content=STATE_BEFORE_ACTION_CONTENT,
            roles=STATE_BEFORE_ACTION_ROLES,
            project_id=None,
        )

    if get_skill_content(EFFICIENT_RAG_NAME) != EFFICIENT_RAG_CONTENT:
        upsert_skill(
            name=EFFICIENT_RAG_NAME,
            description=EFFICIENT_RAG_DESCRIPTION,
            content=EFFICIENT_RAG_CONTENT,
            roles=EFFICIENT_RAG_ROLES,
            project_id=None,
        )


__all__ = ["seed_builtin_skills"]
