# Agent Context Efficiency & Smart Handoff — Implementation Plan

## Overview

This plan addresses efficiency and context management improvements for the TDD Agentic multi-agent system. Goals:

1. **Reduce per-turn token cost** by 30-50% on long runs
2. **Eliminate context duplication** across agent handoffs
3. **Enable smart agent-to-agent communication** with structured, compact handoffs
4. **Keep checkpoints lean** for reliable resume-from-checkpoint

---

## Phase 1: Quick Wins (existing patterns, no new files)

### 1A. Trim AI messages on checkpoint write

**Current problem**: `add_messages_trimmed` in `message_reducer.py` only trims human turns. AI messages (tool results, AI responses) accumulate unbounded in the checkpoint, bloating Postgres and making resume fragile.

**File**: `backend/agents/message_reducer.py`

```python
def trim_checkpoint_messages(
    messages: list,
    *,
    max_human: int | None = None,
    max_ai: int | None = None,
) -> list:
    """Keep first human, recent humans, bounded AI messages."""
    if max_human is None:
        from backend.config import get_settings
        max_human = get_settings().checkpoint_max_human_messages
    if max_ai is None:
        from backend.config import get_settings
        max_ai = get_settings().checkpoint_max_ai_messages  # NEW setting

    if not messages or max_human < 1:
        return []

    humans = [m for m in messages if getattr(m, "type", None) == "human"]
    ai_msgs = [m for m in messages if getattr(m, "type", None) != "human"]

    if len(humans) <= max_human:
        keep_humans = humans
    else:
        first = humans[0]
        tail = humans[-(max_human - 1):]
        keep_humans = [first, *[m for m in tail if m is not first]]

    # Trim AI messages: keep first AI msg + recent tail
    if max_ai is not None and len(ai_msgs) > max_ai:
        first_ai = ai_msgs[0]
        tail_ai = ai_msgs[-(max_ai - 1):]
        keep_ai = [first_ai, *[m for m in tail_ai if m is not first_ai]]
    else:
        keep_ai = ai_msgs

    return keep_humans + keep_ai
```

**Config**: Add `checkpoint_max_ai_messages` to `backend/config.py` (default: 10).

**Impact**: Each turn adds ~2 AI messages (AIMessage + ToolMessages). Over 100 turns, that's ~200 unbounded messages. Capping at 10 reduces checkpoint size by ~95% for long runs.

### 1B. Remove dead SystemState fields

**Current problem**: `pending_questions` and `human_responses` in `state.py` are legacy fields retained for older checkpoints but never written by current nodes. They still get serialized into every checkpoint.

**File**: `backend/agents/state.py`

```python
# Remove or mark as deprecated:
# pending_questions: Annotated[list[str], operator.add] = Field(default_factory=list)
# human_responses: Annotated[list[str], operator.add] = Field(default_factory=list)
# Add: _DEPRECATED = True marker or migrate existing checkpoints
```

**Impact**: 50-100 bytes per checkpoint. Small but free savings.

### 1C. Single canonical project goal

**Current problem**: In `runner.py:_build_specialist_input`, when `project_context` is empty, the first human message is relabeled as `[original project goal]`. In `supervisor.py:_build_system_prompt`, `project_context` is set to the same goal string. And on `start_run` in `agents.py`, `project_context` defaults to `goal`. So the goal is triplicated across fields.

**Fix in `api/routes/agents.py:start_run`**:

```python
# Instead of:
# project_context=payload.project_context or payload.goal
# Use:
initial = SystemState(
    project_id=payload.project_id,
    project_context=payload.goal[:MAX_PROJECT_CONTEXT_CHARS],  # canonical source
    messages=[{"role": "user", "content": payload.goal[:4000]}],
)
```

**Fix in `runner.py:_build_specialist_input`**: Remove the `if not (state.project_context or "").strip():` branch that re-adds the goal from first_human. The goal is already in `project_context`.

**Impact**: Eliminates 500-1500 tokens of goal duplication per specialist invocation.

---

## Phase 2: Structured Handoff Protocol

### 2A. Compact handoff format

**Current problem**: PM→agent handoffs use `[from project_manager → X]\n{json}\n<natural-language instructions>` where instructions repeat ticket content already in the DB.

**New format**: `backend/agents/handoff.py`

```python
"""Structured inter-agent handoff protocol.

Replaces the natural-language handoff strings with a compact,
machine-parseable format that contains only what the next agent
needs to look up.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from enum import Enum

class Phase(str, Enum):
    RESEARCH = "research"
    BACKEND_PLANNING = "backend_planning"
    FRONTEND_PLANNING = "frontend_planning"
    IMPLEMENT = "implement"
    REVIEW = "review"
    QA = "qa"

@dataclass(frozen=True)
class Handoff:
    """Compact structured handoff between agents."""
    target: str
    phase: Phase
    ticket_ids: tuple[str, ...] = ()
    subtask_ids: tuple[str, ...] = ()
    # One-line intent replacing verbose instructions.
    # e.g. "Plan backend subtasks for auth ticket."
    intent: str = ""
    # Cross-agent context: references to prior agent outputs.
    # Points to agent_logs entries, not full text.
    context_refs: tuple[str, ...] = ()
    # Flags for the receiving agent.
    flags: tuple[str, ...] = ()  # e.g. ("requires_research", "has_unanswered_questions")

    def to_message(self) -> str:
        """Serialize to a compact message format for the agent."""
        meta = {
            "t": self.target,
            "p": self.phase.value,
        }
        if self.ticket_ids:
            meta["tik"] = self.ticket_ids
        if self.subtask_ids:
            meta["sid"] = self.subtask_ids
        if self.flags:
            meta["f"] = self.flags
        header = json.dumps(meta, separators=(",", ":"))
        if self.intent:
            return f"[from project_manager → {self.target}]\n{header}\n{self.intent}"
        return f"[from project_manager → {self.target}]\n{header}"

    @classmethod
    def from_routing_decision(cls, next_agent: str, phase: str, instructions: str, ticket_ids: list[str]) -> "Handoff":
        """Create a Handoff from a RoutingDecision."""
        return cls(
            target=next_agent,
            phase=Phase(phase),
            ticket_ids=tuple(ticket_ids),
            intent=instructions[:200],  # Cap intent at 200 chars
        )
```

### 2B. Smart context lookup via context_refs

**New feature**: Agents can reference prior agent outputs by ID rather than receiving full summaries.

**File**: `backend/agents/context_store.py` (NEW)

```python
"""Transient context store for inter-agent communication.

Agents write compact outputs here; other agents can reference them
by ID. Reduces checkpoint size because full summaries don't need
to be in messages.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time

@dataclass
class ContextEntry:
    agent: str
    kind: str  # "research_findings", "lead_plan", "dev_summary", "qa_report"
    ticket_ids: tuple[str, ...]
    summary: str  # max 500 chars
    timestamp: float = field(default_factory=time.time)
    # Full output is stored separately; this is just the index.
    output_key: str = ""  # key into agent_logs or workspace

class ContextStore(dict):
    """In-memory store for agent context references.

    Each entry is a pointer, not a full summary. Agents that need
    the full context call rag_query or read from workspace.
    """
    MAX_ENTRIES = 20
    MAX_SUMMARY_CHARS = 500

    def add(self, agent: str, kind: str, ticket_ids: tuple[str, ...], summary: str, output_key: str = "") -> str:
        """Add a context entry, return its ID."""
        entry_id = f"ctx_{len(self) + 1}"
        self[entry_id] = ContextEntry(
            agent=agent,
            kind=kind,
            ticket_ids=ticket_ids,
            summary=summary[:self.MAX_SUMMARY_CHARS],
            output_key=output_key,
        )
        if len(self) > self.MAX_ENTRIES:
            oldest = next(iter(self))
            del self[oldest]
        return entry_id
```

**Handoff message format with context_refs**:
```
[from project_manager → backend_dev]
{"t":"backend_dev","p":"implement","tik":["uuid1","uuid2"],"c":["ctx_1"],"f":["has_research"]}
Start backend auth subtask. Research findings: ctx_1
```

**Impact**: Instead of 2000-char handoff summary, agent gets a 2-char reference ID + 200-char intent. If they need details, they call `rag_query` with the context_ref.

### 2C. Agent-to-agent direct handoff

**Current problem**: All agents route through the PM. Researcher findings → PM → Lead → PM → Dev. Each hop adds token cost.

**New pattern**: Cross-agent reference links. When the Researcher finishes, it writes findings to workspace + RAG with a context_ref. The PM's handoff to the Lead includes the context_ref. The Lead can optionally reference it directly in its own handoff to the Dev without re-summarizing.

```
Researcher → [writes to RAG] → ctx_3: "Auth uses JWT + refresh tokens"
PM → Lead handoff: {"tik":["uuid1"],"c":["ctx_3"]}
Lead → Dev handoff: {"tik":["uuid1","uuid2"],"c":["ctx_3","ctx_5"]}
```

**This avoids the Researcher→PM→Lead token tax** where the PM re-summarizes the Researcher's work in natural language.

---

## Phase 3: Smart Context Injection

### 3A. Static system prompt cache

**Current problem**: Every specialist subgraph build calls `inject_skills(base_system_prompt, role)` which loads the full prompt string (6KB for PM, 8KB for leads with RITE appendix) on every single invocation.

**Fix**: Cache the base prompts and only inject dynamic fragments (skills, project_context, active_ticket_id) at runtime.

```python
# backend/agents/prompts.py
from functools import lru_cache

@lru_cache(maxsize=1)
def get_cached_system_prompt(role: str) -> str:
    """Cache the static portion of the system prompt.
    
    Only called once at app startup. Dynamic parts (skills, project_context)
    are appended at build time.
    """
    if role == "project_manager":
        return PROJECT_MANAGER_SYSTEM
    elif role == "lead":
        return LEAD_SYSTEM
    # ... etc

@lru_cache(maxsize=1)
def get_cached_lead_appendix() -> str:
    return _RITE_CONTRACT + _LEAD_TOOL_CONTRACT
```

**Impact**: No measurable token savings (prompts were always loaded from module), but eliminates repeated string concatenation and skill loading overhead.

### 3B. Skill loading optimization

**Current problem**: `inject_skills` loads all skills for every role on every turn. Even though it only injects names/descriptions (not full bodies), this is still unnecessary if no new skills exist since the last turn.

**Fix**: In `skills/loader.py`, add a change detection mechanism:

```python
def inject_skills(base_prompt: str, role: str) -> str:
    """Append skills only if they've changed since last injection."""
    skills = get_skills_for_role(role)
    if not skills:
        return base_prompt

    # Cache the last injected skill set per role
    cache_key = f"skills_{role}"
    current_hash = hash(frozenset(s.get("name") for s in skills))
    if getattr(inject_skills, "_cache", None).get(cache_key) == current_hash:
        return base_prompt  # No change, skip injection

    # ... proceed with normal injection ...
    inject_skills._cache[cache_key] = current_hash
    return result
```

### 3C. RAG-based skill loading

**Current problem**: Skills are injected into the system prompt up to 8000 chars. For projects with many skills, this is a significant fraction of the context budget.

**Fix**: Instead of injecting full skill content, inject a compact skill index and instruct the agent to `rag_query` the specific skill when needed.

```python
# Current: inject_skills appends all skill names + descriptions
# New: also inject a rag_query hint for each skill
SKILL_INDEX = """
Assigned skills (call rag_query with the skill name for full content):
- auth-patterns: JWT + refresh token auth flow
- nextjs-best-practices: App Router patterns, data fetching
...
Use rag_query(skill_name) to load full SKILL.md content when working on relevant tasks.
"""
```

---

## Phase 4: Context Compression

### 4A. Cross-agent context deduplication

**Current problem**: The same research findings, RITE contract, and ticket requirements are repeated across every PM→specialist handoff and every specialist summary.

**Fix**: The `ContextStore` (Phase 2B) serves as the de-duplication layer. Handoffs contain only refs:

```
# Before (duplicate content):
[from project_manager → backend_dev]
{"ticket_ids":["uuid1"]}
"Implement JWT auth. The auth system should use JWT tokens with 15min expiry, 
refresh tokens with 7day expiry, stored in HTTP-only cookies. See research 
findings about using python-jose for JWT handling..."

# After (compact reference):
[from project_manager → backend_dev]
{"t":"backend_dev","p":"implement","tik":["uuid1"],"c":["ctx_3","ctx_7"]}
"Implement JWT auth per ctx_3 (research) and ctx_7 (tech spec)."
```

### 4B. Ticket tool response shaping

**Current problem**: `get_ticket` returns full subtask trees with RITE test cases on every call. PM only needs ticket status + subtask count for routing.

**Fix**: Add `detail` parameter variants:

```python
# backend/tools/ticket_tools.py
def list_tickets_compact(project_id: str) -> list:  # Already exists, returns summary
    """Return ticket roster without RITE trees."""
    ...

def get_ticket_summary(ticket_id: str) -> dict:  # NEW
    """Return ticket without subtask details."""
    ...
```

PM uses `list_tickets_compact` + `get_ticket_summary` for routing decisions, only calling `get_ticket(detail='full')` when actually dispatching a developer.

---

## Implementation Priority

| Priority | Change | Effort | Token Savings (per turn) |
|----------|--------|--------|------------------------|
| P1 | Trim AI messages on write | 30 min | 500-2000 chars |
| P2 | Remove dead SystemState fields | 15 min | 50-100 bytes |
| P3 | Single canonical goal | 30 min | 500-1500 tokens |
| P4 | Structured handoff protocol | 2 hrs | 200-500 tokens/handoff |
| P5 | Context store (context_refs) | 2 hrs | 500-1500 tokens/handoff |
| P6 | RAG-based skill loading | 1 hr | 500-2000 tokens (many skills) |
| P7 | Ticket tool response shaping | 1 hr | 300-1000 tokens/PM turn |
| P8 | Cross-agent reference links | 3 hrs | 200-500 tokens/handoff |

---

## Summary

The most impactful changes are already partially implemented (condensed human history, focused specialist input, PM not persisting AI/tool messages). The remaining improvements focus on:

1. **Trimming on write** (not just on read) to prevent checkpoint bloat
2. **Replacing natural-language handoffs** with compact structured refs
3. **Eliminating goal duplication** across the three fields
4. **Lazy skill/context loading** instead of full injection every turn

These changes preserve the existing architecture while making it significantly more efficient for long-running multi-agent projects.
