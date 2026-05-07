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


def _registry_path() -> Path:
    settings = get_settings()
    base = settings.workspace_root / "_skills"
    base.mkdir(parents=True, exist_ok=True)
    return base / "registry.json"


_lock = threading.Lock()


def _load_raw() -> dict[str, Any]:
    path = _registry_path()
    if not path.exists():
        return {"skills": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"skills": {}}


def _save_raw(data: dict[str, Any]) -> None:
    _registry_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def upsert_skill(
    name: str,
    description: str,
    content: str,
    roles: list[str],
    project_id: str | None = None,
) -> dict[str, Any]:
    """Create or update a skill and persist its content to disk."""
    settings = get_settings()
    skill_dir = settings.workspace_root / "_skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

    with _lock:
        data = _load_raw()
        data["skills"][name] = {
            "name": name,
            "description": description,
            "roles": list(roles),
            "project_id": project_id,
            "path": str((skill_dir / "SKILL.md").as_posix()),
        }
        _save_raw(data)
    return data["skills"][name]


def list_skills() -> list[dict[str, Any]]:
    return list(_load_raw().get("skills", {}).values())


def get_skills_for_role(role: str) -> list[dict[str, Any]]:
    return [s for s in list_skills() if role in (s.get("roles") or [])]


def get_skill_content(name: str) -> str | None:
    skills = _load_raw().get("skills", {})
    info = skills.get(name)
    if info is None:
        return None
    path = Path(info["path"])
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")
