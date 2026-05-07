"""Skills system: per-role micro-prompts loaded into agents at runtime."""
from backend.agents.skills.loader import inject_skills
from backend.agents.skills.registry import (
    get_skill_content,
    get_skills_for_role,
    list_skills,
    upsert_skill,
)

__all__ = [
    "inject_skills",
    "get_skill_content",
    "get_skills_for_role",
    "list_skills",
    "upsert_skill",
]
