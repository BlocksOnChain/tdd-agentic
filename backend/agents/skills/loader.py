"""Skill injection helper used by every agent's prompt builder."""
from __future__ import annotations

from backend.agents.skills.registry import get_skill_content, get_skills_for_role


def inject_skills(base_prompt: str, role: str, max_chars: int = 8000) -> str:
    """Append the SKILL.md contents of every skill assigned to ``role``."""
    skills = get_skills_for_role(role)
    if not skills:
        return base_prompt

    parts: list[str] = []
    used = 0
    for s in skills:
        content = get_skill_content(s["name"])
        if not content:
            continue
        block = f"\n\n=== SKILL: {s['name']} ===\n{content}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    if not parts:
        return base_prompt
    return f"{base_prompt}\n\n--- ASSIGNED SKILLS ---{''.join(parts)}"
