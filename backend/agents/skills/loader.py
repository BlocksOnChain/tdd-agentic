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

    Change detection: if the skill set for ``role`` hasn't changed since the
    last call, return ``base_prompt`` unchanged (saves string concatenation).
    """
    from backend.config import get_settings

    skills = get_skills_for_role(role)
    if not skills:
        return base_prompt

    # Compute a stable hash of the current skill set.
    current_hash = hash(frozenset(s.get("name", "") for s in skills))
    cache_key = f"skills_{role}"
    if cache_key in _inject_cache and _inject_cache[cache_key][0] == current_hash:
        return base_prompt  # No change — skip injection.

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
