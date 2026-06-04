"""Agent execution host facts for prompts and workspace AGENTS.md.

``shell_run`` and ``run_tests`` run inside the orchestrator backend container
(Dockerfile target ``slim``: python:3.12-slim + Node.js), not on the host OS
and not inside per-project docker-compose services unless explicitly wired.
"""
from __future__ import annotations

import platform
import shutil
from functools import lru_cache
from pathlib import Path


def _parse_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"')
    return out


@lru_cache(maxsize=1)
def detect_agent_runtime() -> dict[str, str | bool]:
    """Best-effort facts about the process that runs agent tools."""
    os_release = _parse_os_release()
    arch = platform.machine() or "unknown"
    os_id = os_release.get("ID", "")
    version_id = os_release.get("VERSION_ID", "")
    distro_tag = f"{os_id}-{version_id}" if os_id and version_id else os_id or "unknown"

    return {
        "arch": arch,
        "platform": platform.system(),
        "os_pretty": os_release.get("PRETTY_NAME", "unknown"),
        "os_id": os_id,
        "os_version_id": version_id,
        "distro_tag": distro_tag,
        "node_path": shutil.which("node") or "",
        "npm_path": shutil.which("npm") or "",
        "docker_cli": shutil.which("docker") is not None,
        "mongod_path": shutil.which("mongod") or "",
    }


def _mongodb_memory_server_hints(arch: str, os_id: str) -> list[str]:
    hints: list[str] = [
        "mongodb-memory-server downloads a MongoDB **binary** at runtime; npm install "
        "only installs the wrapper. A 403 from fastdl.mongodb.org usually means that "
        "distro/arch/version combination does not exist — not a network flake.",
    ]
    if platform.system().lower() != "linux":
        return hints

    arch_l = arch.lower()
    os_l = os_id.lower()
    if arch_l in ("aarch64", "arm64") and os_l in ("debian", ""):
        hints.append(
            "On Linux arm64 under Debian, MongoDB publishes **no** official arm64 Debian "
            "builds. Do not retry random versions in shell_run. Use Ubuntu binaries instead: "
            "set MONGOMS_DISTRO=ubuntu-22.04 (and MONGOMS_VERSION>=7.0.3 on Debian 12+), "
            "or rely on the platform env vars already set in the orchestrator container."
        )
    return hints


@lru_cache(maxsize=1)
def get_agent_runtime_prompt_section() -> str:
    """Compact block appended to every agent system prompt."""
    r = detect_agent_runtime()
    arch = str(r["arch"])
    hints = _mongodb_memory_server_hints(arch, str(r["os_id"]))

    lines = [
        "=== AGENT EXECUTION ENVIRONMENT (authoritative) ===",
        "All shell_run / run_tests commands run in the **orchestrator backend container**,",
        "not on your laptop and not inside project docker-compose services by default.",
        "",
        "Detected this process:",
        f"  - OS: {r['os_pretty']} (id={r['distro_tag']})",
        f"  - CPU arch: {arch}",
        f"  - Node: {r['node_path'] or '(not on PATH)'}",
        f"  - npm: {r['npm_path'] or '(not on PATH)'}",
        f"  - mongod on PATH: {'yes' if r['mongod_path'] else 'no'}",
        f"  - docker CLI: {'yes' if r['docker_cli'] else 'no — cannot docker compose up from agents'}",
        "",
        "Implications:",
        "  - Prefer apt/npm packages that support this OS+arch; do not assume macOS or Ubuntu",
        "    binaries exist for Debian arm64.",
        "  - Tools that download OS-specific binaries (mongodb-memory-server, Playwright",
        "    browsers, etc.) may need MONGOMS_* / PLAYWRIGHT_* env overrides — npm alone is",
        "    not enough.",
        "  - For databases in tests: use injected fakes in unit tests, a platform-provided",
        "    URI, or project compose on a reachable host — not endless shell download loops.",
    ]
    for h in hints:
        lines.append(f"  - {h}")
    lines.append(
        "  - Read workspace AGENTS.md § Agent execution environment for the canonical spec."
    )
    return "\n".join(lines)


def get_agent_runtime_agents_md_section() -> str:
    """Markdown section seeded into every new project AGENTS.md."""
    r = detect_agent_runtime()
    arch = str(r["arch"])
    mongodb_note = ""
    if platform.system().lower() == "linux" and arch.lower() in ("aarch64", "arm64"):
        mongodb_note = (
            "\n- **MongoDB tests (`mongodb-memory-server`)**: On Debian arm64 there is no "
            "official MongoDB Debian arm64 tarball. The orchestrator sets "
            "`MONGOMS_DISTRO=ubuntu-22.04` so downloads work. Do not keep changing versions "
            "in `shell_run` when stderr shows `fastdl.mongodb.org` + `debian` + 403.\n"
        )

    return f"""## Agent execution environment

Commands from `shell_run` and `run_tests` execute in the **tdd-agentic backend container**
(`python:3.12-slim` + Node.js LTS), mounted at `/app/workspace/<project_id>/`.

| Property | Typical value |
|----------|----------------|
| OS | Debian (from `python:3.12-slim`; exact release varies) |
| Architectures | `aarch64` (Apple Silicon / ARM servers) or `x86_64` |
| Node / npm | Installed globally in the backend image |
| Docker CLI | **Not** available inside the agent container |
| `mongod` | Not pre-installed unless added to the backend image |

Detected in this deployment: **{r["os_pretty"]}**, arch **{arch}**.
{mongodb_note}
### Native binaries vs npm

- `npm install` only adds JavaScript; packages like **mongodb-memory-server** and **Playwright**
  still download large **platform-specific binaries**. If that download 403s or times out,
  fix distro/arch config (`MONGOMS_DISTRO`, etc.) or change the test strategy — do not loop
  on `npm install` or version bumps alone.

### What not to do

- Do not assume project `docker-compose.yml` services are reachable from agent commands unless
  documented (agents are not on the project compose network by default).
- Do not run `docker compose` from agents (no Docker socket in the backend container).
- After the same binary-download error **3+** times, mark the subtask **blocked** and escalate.
"""
