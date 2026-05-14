
"""Skill registry.

A skill is a small markdown file describing a focused capability (a library,
a pattern, a workflow). Skills are persisted in JSON form here and indexed
to RAG so retrieval-time lookup works too.

Roles map: project_manager, researcher, backend_lead, frontend_lead,
backend_dev, frontend_dev, devops, qa.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from backend.config import get_settings

ALLOWED_SKILL_ROLES = frozenset(
    {
        "project_manager",
        "researcher",
        "backend_lead",
        "frontend_lead",
        "backend_dev",
        "frontend_dev",
        "devops",
        "qa",
    }
)

_lock = threading.Lock()


def _project_skill_root(project_id: str | None) -> Path:
    settings = get_settings()
    if project_id:
        return settings.workspace_root / project_id / "_skills"
    return settings.workspace_root / "_skills"


def _registry_path(project_id: str | None) -> Path:
    root = _project_skill_root(project_id)
    root.mkdir(parents=True, exist_ok=True)
    return root / "registry.json"


def _load_raw(project_id: str | None) -> dict[str, Any]:
    path = _registry_path(project_id)
    if not path.exists():
        return {"skills": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"skills": {}}


def _save_raw(project_id: str | None, data: dict[str, Any]) -> None:
    _registry_path(project_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def upsert_skill(
    name: str,
    description: str,
    content: str,
    roles: list[str],
    project_id: str | None = None,
) -> dict[str, Any]:
    """Create or update a skill and persist its content to disk."""
    invalid = [role for role in roles if role not in ALLOWED_SKILL_ROLES]
    if invalid:
        raise ValueError(
            "Invalid skill roles: "
            f"{', '.join(invalid)}. Allowed: {', '.join(sorted(ALLOWED_SKILL_ROLES))}."
        )

    skill_dir = _project_skill_root(project_id) / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    with _lock:
        data = _load_raw(project_id)
        data["skills"][name] = {
            "name": name,
            "description": description,
            "roles": list(roles),
            "project_id": project_id,
            "path": str((skill_dir / "SKILL.md").as_posix()),
        }
        _save_raw(project_id, data)
    return data["skills"][name]


def list_skills(project_id: str | None = None) -> list[dict[str, Any]]:
    if project_id:
        return list(_load_raw(project_id).get("skills", {}).values())
    settings = get_settings()
    root = settings.workspace_root
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for project_dir in sorted(root.iterdir()):
        if not project_dir.is_dir():
            continue
        rows.extend(_load_raw(project_dir.name).get("skills", {}).values())
    legacy = _load_raw(None).get("skills", {})
    rows.extend(legacy.values())
    return rows


def get_skills_for_role(role: str, project_id: str | None = None) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for skill in list_skills(None):
        if role in (skill.get("roles") or []):
            by_name[str(skill.get("name") or "")] = skill
    if project_id:
        for skill in list_skills(project_id):
            if role in (skill.get("roles") or []):
                by_name[str(skill.get("name") or "")] = skill
    return [skill for name, skill in by_name.items() if name]


def get_skill_content(name: str, project_id: str | None = None) -> str | None:
    skills = _load_raw(project_id).get("skills", {})
    info = skills.get(name)
    if info is None and project_id is not None:
        info = _load_raw(None).get("skills", {}).get(name)
    if info is None:
        return None
    path = Path(info["path"])
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")
