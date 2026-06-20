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
    """Write or overwrite a file inside the project workspace.

    USE WHEN: You need to create or update source, config, test, or doc files for the subtask.
    USE WHEN: Parent directories do not exist yet — they are created automatically.
    AVOID WHEN: You only need to read or inspect a file — use fs_read instead.
    AVOID WHEN: The path escapes the project workspace (absolute paths outside the tree are rejected).

    RETURNS: {ok: true, path: "<relative path>"}
    ON ERROR (path escape): raises ValueError before write.
    """
    full = _resolve(project_id, path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return json.dumps({"ok": True, "path": str(full.relative_to(_project_root(project_id)))})


@tool
async def fs_read(project_id: str, path: str) -> str:
    """Read a text file inside the project workspace.

    USE WHEN: You need file contents before editing, debugging, or implementing a change.
    USE WHEN: You want to verify what fs_write produced or inspect an existing module.
    AVOID WHEN: You only need filenames or directory layout — use fs_list instead.
    AVOID WHEN: You already read the same path this turn — reuse the prior result.

    Content is capped at FS_READ_MAX_BYTES (default 32_768); large files return truncated=true.
    RETURNS: {ok: true, content: "...", truncated: bool, bytes_read: int, bytes_total: int}
    ON ERROR (not found): {ok: false, error: "not found"}
    ON ERROR (path escape): raises ValueError before read.
    """
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
    """List immediate children of a directory in the project workspace.

    USE WHEN: You need to discover project layout, find a file path, or confirm a directory exists.
    USE WHEN: You are orienting in an unfamiliar repo before reading specific files.
    AVOID WHEN: You already know the exact file path — use fs_read directly.
    AVOID WHEN: You need recursive tree listing — list subdirectories one level at a time.

    RETURNS: {ok: true, entries: [{name, is_dir, size}, ...]} sorted by name.
    ON ERROR (not found): {ok: false, error: "not found"}
    ON ERROR (path escape): raises ValueError before list.
    """
    full = _resolve(project_id, path)
    if not full.exists():
        return json.dumps({"ok": False, "error": "not found"})
    entries = []
    for child in sorted(full.iterdir()):
        entries.append({"name": child.name, "is_dir": child.is_dir(), "size": child.stat().st_size})
    return json.dumps({"ok": True, "entries": entries})


@tool
async def fs_delete(project_id: str, path: str) -> str:
    """Delete a file or empty directory inside the project workspace.

    USE WHEN: You need to remove a file you created or an obsolete empty directory.
    AVOID WHEN: The directory is non-empty — fs_delete only removes empty dirs (rmdir).
    AVOID WHEN: You can achieve the goal by editing in place — prefer fs_write over delete+recreate.

    RETURNS: {ok: true}
    ON ERROR (not found): {ok: false, error: "not found"}
    ON ERROR (non-empty directory): subprocess/OS error surfaced as exception (not JSON).
    ON ERROR (path escape): raises ValueError before delete.
    """
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
    """Run a shell command with cwd set to the project workspace.

    USE WHEN: You need to install deps, run linters, build artifacts, or diagnose environment issues.
    USE WHEN: A one-off command is needed that is not the primary test verification step.
    AVOID WHEN: You are verifying tests pass/fail for subtask completion — use run_tests instead (that is the verification gate).
    AVOID WHEN: fs_read/fs_write/fs_list can answer the question without spawning a process.

    stdout/stderr are truncated to the last SHELL_OUTPUT_MAX_CHARS (default 8_000) characters.
    RETURNS: {ok: bool, exit_code: int, stdout: "...", stderr: "...", stdout_truncated: bool, stderr_truncated: bool}
    ON ERROR (timeout): {ok: false, error: "timeout after <N>s", command: "..."}
    ON ERROR (spawn failure): {ok: false, error: "<exception repr>"}
    """
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
    """Run the test suite (or a focused test command) as the subtask verification gate.

    USE WHEN: You have implemented or fixed code and need pass/fail signal before marking work done.
    USE WHEN: TDD red-green-refactor — run after each meaningful change to confirm behavior.
    AVOID WHEN: You only need to install packages or inspect the environment — use shell_run instead.
    AVOID WHEN: You have not written or updated tests yet and expect failure — implement first, then run.

    Default command stops after the first failure (--maxfail=1). Timeout is 300s (longer than shell_run).
    Same JSON shape as shell_run; ok reflects exit_code == 0.
    RETURNS: {ok: bool, exit_code: int, stdout: "...", stderr: "...", stdout_truncated: bool, stderr_truncated: bool}
    ON ERROR (timeout): {ok: false, error: "timeout after 300s", command: "..."}
    """
    return await shell_run.ainvoke(
        {"project_id": project_id, "command": command, "timeout_seconds": 300}
    )


CODE_TOOLS = [fs_write, fs_read, fs_list, fs_delete, shell_run, run_tests]
