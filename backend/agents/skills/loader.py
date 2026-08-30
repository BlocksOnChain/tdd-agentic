"""Skill injection helper used by every agent's prompt builder.

Uses change detection to skip injection when skills haven't changed,
avoiding unnecessary string concatenation on every turn.
"""
from __future__ import annotations

from backend.agents.skills.registry import get_skills_for_role


# Cache the last injected skill set per role — keyed by frozenset of names.
# On every call, compare the current set of skill names against the cached hash.
_inject_cache: dict[str, tuple[int, str]] = {}

_ALWAYS_INLINE_SKILLS = frozenset(
    {
        # User preference: caveman mode always active from start.
        "caveman",
    }
)


def inject_skills(base_prompt: str, role: str, max_chars: int | None = None) -> str:
    """Append a compact index of skills assigned to ``role``.

    Full ``SKILL.md`` bodies are available via ``rag_query``; only names and
    descriptions are inlined to keep system prompts small.

    Change detection: if neither the skill set for ``role`` nor ``base_prompt``
    has changed since the last call, the previously assembled prompt is returned
    from cache (saves re-reading skill bodies and re-joining strings). The
    injected content itself is never skipped — every turn gets the full index.
    """
    from backend.config import get_settings

    skills = get_skills_for_role(role)
    if not skills:
        return base_prompt

    # Compute a stable hash of the current skill set *and* the prompt we are
    # appending to — the cached value is the fully assembled prompt, so it is
    # only reusable when both halves are unchanged.
    current_hash = hash((frozenset(s.get("name", "") for s in skills), base_prompt))
    cache_key = f"skills_{role}"
    cached = _inject_cache.get(cache_key)
    if cached is not None and cached[0] == current_hash:
        return cached[1]  # No change — reuse the assembled prompt.

    budget = max_chars if max_chars is not None else get_settings().skill_inject_max_chars
    lines: list[str] = [
        "Assigned skills (call rag_query with the skill name for full SKILL.md content):"
    ]
    used = len(lines[0])
    inline_skill_names: list[str] = []
    for s in skills:
        name = s.get("name", "")
        desc = (s.get("description") or "").strip()
        line = f"- {name}: {desc}" if desc else f"- {name}"
        if used + len(line) + 1 > budget:
            lines.append("- …(additional skills omitted; use rag_query)")
            break
        lines.append(line)
        used += len(line) + 1
        if name in _ALWAYS_INLINE_SKILLS:
            inline_skill_names.append(name)
    lines.append("")
    lines.append("Use rag_query(skill_name) to load full SKILL.md content when working on relevant tasks.")

    # Always-inline: some skills must materially change agent behavior, not just be discoverable.
    # Keep bounded so we don't blow prompt budgets.
    inline_blocks: list[str] = []
    if inline_skill_names:
        from backend.agents.skills.registry import get_skill_content

        for name in inline_skill_names:
            body = (get_skill_content(name) or "").strip()
            if not body:
                continue
            # Hard cap per inline skill to avoid runaway prompt growth.
            inline_blocks.append(f"\n--- ALWAYS-ON SKILL: {name} ---\n{body[:1200]}")

    result = (
        f"{base_prompt}\n\n--- ASSIGNED SKILLS ---\n"
        + "\n".join(lines)
        + ("\n".join(inline_blocks) if inline_blocks else "")
    )

    # Update cache.
    _inject_cache[cache_key] = (current_hash, result)
    return result
