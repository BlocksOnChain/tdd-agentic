"""Filesystem + subprocess tools for developer agents.

All paths are resolved relative to the per-project workspace root and
constrained to that subtree to prevent escapes. Subprocess calls have a
hard wall-clock timeout.
"""
from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path

from langchain_core.tools import tool

from backend.config import get_settings


def _project_root(project_id: str) -> Path:
    settings = get_settings()
    root = settings.workspace_root / project_id
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _resolve(project_id: str, rel_path: str) -> Path:
    root = _project_root(project_id)
    candidate = (root / rel_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"path '{rel_path}' escapes project workspace")
    return candidate


@tool
async def fs_write(project_id: str, path: str, content: str) -> str:
    """Write a file inside the project workspace. Creates parent dirs."""
    full = _resolve(project_id, path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return json.dumps({"ok": True, "path": str(full.relative_to(_project_root(project_id)))})


@tool
async def fs_read(project_id: str, path: str) -> str:
    """Read a file inside the project workspace (bounded by FS_READ_MAX_BYTES)."""
    settings = get_settings()
    full = _resolve(project_id, path)
    if not full.exists():
        return json.dumps({"ok": False, "error": "not found"})
    raw = full.read_bytes()
    cap = settings.fs_read_max_bytes
    truncated = len(raw) > cap
    text = raw[:cap].decode("utf-8", errors="replace")
    return json.dumps(
        {
            "ok": True,
            "content": text,
            "truncated": truncated,
            "bytes_read": min(len(raw), cap),
            "bytes_total": len(raw),
        }
    )


@tool
async def fs_list(project_id: str, path: str = ".") -> str:
    """List files and directories inside the project workspace."""
    full = _resolve(project_id, path)
    if not full.exists():
        return json.dumps({"ok": False, "error": "not found"})
    entries = []
    for child in sorted(full.iterdir()):
        entries.append({"name": child.name, "is_dir": child.is_dir(), "size": child.stat().st_size})
    return json.dumps({"ok": True, "entries": entries})


@tool
async def fs_delete(project_id: str, path: str) -> str:
    """Delete a file or empty directory inside the project workspace."""
    full = _resolve(project_id, path)
    if not full.exists():
        return json.dumps({"ok": False, "error": "not found"})
    if full.is_dir():
        full.rmdir()
    else:
        full.unlink()
    return json.dumps({"ok": True})


@tool
async def shell_run(project_id: str, command: str, timeout_seconds: int = 120) -> str:
    """Run a shell command inside the project workspace. Captures stdout/stderr."""
    cwd = _project_root(project_id)
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            proc.kill()
            return json.dumps(
                {"ok": False, "error": f"timeout after {timeout_seconds}s", "command": command}
            )
        cap = get_settings().shell_output_max_chars
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        return json.dumps(
            {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": out[-cap:],
                "stderr": err[-cap:],
                "stdout_truncated": len(out) > cap,
                "stderr_truncated": len(err) > cap,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": repr(exc)})


@tool
async def run_tests(project_id: str, command: str = "pytest -q --maxfail=1") -> str:
    """Run a focused test command (default stops after the first failure)."""
    return await shell_run.ainvoke(
        {"project_id": project_id, "command": command, "timeout_seconds": 300}
    )


CODE_TOOLS = [fs_write, fs_read, fs_list, fs_delete, shell_run, run_tests]
