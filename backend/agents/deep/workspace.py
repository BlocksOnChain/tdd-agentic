"""Per-project filesystem backend for Deep Agents."""
from __future__ import annotations

from deepagents.backends.filesystem import FilesystemBackend

from backend.config import get_settings


def project_filesystem_backend(project_id: str) -> FilesystemBackend:
    """Workspace rooted at ``settings.workspace_root / project_id`` with virtual paths."""
    root = (get_settings().workspace_root / project_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return FilesystemBackend(root_dir=str(root), virtual_mode=True)
