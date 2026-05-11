"""Skill injection helper used by every agent's prompt builder."""
from __future__ import annotations

from backend.agents.skills.registry import get_skills_for_role


def inject_skills(
    base_prompt: str,
    role: str,
    *,
    project_id: str | None = None,
    max_chars: int | None = None,
) -> str:
    """Append a compact index of skills assigned to ``role``.

    Full ``SKILL.md`` bodies are available via ``rag_query``; only names and
    descriptions are inlined to keep system prompts small.
    """
    from backend.config import get_settings

    skills = get_skills_for_role(role, project_id)
    if not skills:
        return base_prompt

    budget = max_chars if max_chars is not None else get_settings().skill_inject_max_chars
    lines: list[str] = [
        "Assigned skills (call rag_query with the skill name for full SKILL.md content):"
    ]
    used = len(lines[0])
    for s in skills:
        name = s.get("name", "")
        desc = (s.get("description") or "").strip()
        line = f"- {name}: {desc}" if desc else f"- {name}"
        if used + len(line) + 1 > budget:
            lines.append("- …(additional skills omitted; use rag_query)")
            break
        lines.append(line)
        used += len(line) + 1
    return f"{base_prompt}\n\n--- ASSIGNED SKILLS ---\n" + "\n".join(lines)
