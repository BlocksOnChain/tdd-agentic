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
    "backend_lead",
    "frontend_lead",
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


__all__ = ["seed_builtin_skills"]
