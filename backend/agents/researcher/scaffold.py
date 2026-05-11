"""Track auto-scaffolded research docs so PM gates ignore placeholder files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import get_settings


def project_workspace_root(project_id: str) -> Path:
    root = (get_settings().workspace_root / project_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root

_MANIFEST_NAME = ".research_scaffold.json"


def scaffold_manifest_path(project_id: str) -> Path:
    return project_workspace_root(project_id) / _MANIFEST_NAME


def load_scaffolded_paths(project_id: str) -> set[str]:
    path = scaffold_manifest_path(project_id)
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    paths = data.get("paths") if isinstance(data, dict) else None
    if not isinstance(paths, list):
        return set()
    return {str(item) for item in paths if item}


def record_scaffolded_paths(project_id: str, paths: list[str]) -> None:
    if not paths:
        return
    manifest = scaffold_manifest_path(project_id)
    existing = load_scaffolded_paths(project_id)
    merged = sorted(existing | set(paths))
    payload: dict[str, Any] = {"paths": merged}
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_scaffolded_path(project_id: str, rel_path: str) -> bool:
    return rel_path in load_scaffolded_paths(project_id)


def is_placeholder_scaffold_content(rel_path: str, content: str) -> bool:
    return _still_scaffold_placeholder(rel_path, content)


def should_skip_research_rag_index(project_id: str, rel_path: str, content: str) -> bool:
    """True when workspace markdown is still auto-scaffold placeholder text."""
    del project_id
    return is_placeholder_scaffold_content(rel_path, content)


_PLACEHOLDER_MARKERS: dict[str, tuple[str, ...]] = {
    "docs/tech-stack.md": (
        "Document the chosen frontend, backend, database, and test tooling here.",
    ),
    "docs/architecture.md": (
        "Describe the main components, data flow, and deployment shape here.",
    ),
}


def _still_scaffold_placeholder(rel_path: str, content: str) -> bool:
    markers = _PLACEHOLDER_MARKERS.get(rel_path)
    if not markers:
        return False
    normalized = content.strip()
    if not normalized:
        return True
    if not all(marker in normalized for marker in markers):
        return False
    return len(normalized) < 500


def refresh_authored_scaffold_paths(project_id: str) -> list[str]:
    """Drop auto-scaffold markers once a file contains substantive authored content."""
    scaffolded = load_scaffolded_paths(project_id)
    if not scaffolded:
        return []

    root = project_workspace_root(project_id)
    cleared: list[str] = []
    for rel_path in sorted(scaffolded):
        path = root / rel_path
        if not path.is_file():
            cleared.append(rel_path)
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not _still_scaffold_placeholder(rel_path, content):
            cleared.append(rel_path)

    if not cleared:
        return []

    remaining = sorted(scaffolded - set(cleared))
    manifest = scaffold_manifest_path(project_id)
    if remaining:
        payload: dict[str, Any] = {"paths": remaining}
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif manifest.is_file():
        manifest.unlink()
    return cleared


def discard_placeholder_scaffold_files(project_id: str) -> list[str]:
    """Remove placeholder scaffold files so the researcher must author real docs."""
    scaffolded = load_scaffolded_paths(project_id)
    if not scaffolded:
        return []

    root = project_workspace_root(project_id)
    removed: list[str] = []
    for rel_path in sorted(scaffolded):
        path = root / rel_path
        if not path.is_file():
            removed.append(rel_path)
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not _still_scaffold_placeholder(rel_path, content):
            continue
        path.unlink()
        removed.append(rel_path)

    if not removed:
        return []

    remaining = sorted(scaffolded - set(removed))
    manifest = scaffold_manifest_path(project_id)
    if remaining:
        payload: dict[str, Any] = {"paths": remaining}
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif manifest.is_file():
        manifest.unlink()
    return removed
