"""Skill registry.

A skill is a small markdown file describing a focused capability (a library,
a pattern, a workflow). Skills are persisted in JSON form here and indexed
to RAG so retrieval-time lookup works too.

Roles map: project_manager, researcher, lead, coordinator,
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

# Roles that no longer exist. ``backend_lead`` and ``frontend_lead`` were merged
# into a single ``lead``, but registries written before that merge still name
# them — so those skills reach nobody. Remap on read; existing workspaces are
# not reseeded.
_LEGACY_ROLE_ALIASES: dict[str, str] = {
    "backend_lead": "lead",
    "frontend_lead": "lead",
}

# (mtime, size) -> parsed registry. get_skills_for_role is called for every
# agent turn and used to re-read and re-parse the file each time (twice, when
# a skill body was inlined).
_cache: tuple[tuple[float, int], dict[str, Any]] | None = None


def _normalise_roles(roles: Any) -> list[str]:
    out: list[str] = []
    for role in roles or []:
        mapped = _LEGACY_ROLE_ALIASES.get(str(role), str(role))
        if mapped not in out:
            out.append(mapped)
    return out


def _load_raw() -> dict[str, Any]:
    global _cache
    path = _registry_path()
    if not path.exists():
        return {"skills": {}}
    try:
        stat = path.stat()
        stamp = (stat.st_mtime, stat.st_size)
    except OSError:
        stamp = None

    if _cache is not None and stamp is not None and _cache[0] == stamp:
        return _cache[1]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"skills": {}}

    for info in (data.get("skills") or {}).values():
        if isinstance(info, dict):
            info["roles"] = _normalise_roles(info.get("roles"))

    if stamp is not None:
        _cache = (stamp, data)
    return data


def invalidate_registry_cache() -> None:
    global _cache
    _cache = None


def _save_raw(data: dict[str, Any]) -> None:
    _registry_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    invalidate_registry_cache()


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
    """Read a skill's SKILL.md.

    The registry stores an absolute path, resolved when the skill was written.
    That path is baked in at seed time (``/app/workspace/...`` inside the
    container), so it does not resolve when the same registry is read with a
    different ``WORKSPACE_ROOT`` — a dev machine, a test, a relocated volume.
    Prefer the location implied by the current workspace root and fall back to
    the stored path.
    """
    info = _load_raw().get("skills", {}).get(name)
    if info is None:
        return None

    candidates = [get_settings().workspace_root / "_skills" / name / "SKILL.md"]
    stored = info.get("path")
    if stored:
        candidates.append(Path(stored))

    for path in candidates:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except OSError:
            continue
    return None
