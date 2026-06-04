# Agent notes (this workspace)

## Where documentation lives

- **Do not** point agents at `node_modules/` for guides or API references. Vendored packages are not a stable documentation surface and should stay out of prompts, skills, and RAG context.
- Put **project-specific** and **framework** notes under **`docs/`** in this workspace (for example `docs/nextjs.md`, `docs/conventions.md`, `docs/api.md`). The Researcher should author or refresh these and persist them into RAG with `rag_ingest_text`.
- For **canonical** vendor documentation, use **stable HTTPS URLs** (for example [Next.js docs](https://nextjs.org/docs)) and copy only the behavior that matters into `docs/` for offline, version-aligned reference.

## Framework stacks (e.g. Next.js)

Installed behavior is defined by this repo’s `package.json` and lockfile. If stack knowledge may be outdated, prefer `docs/<framework>.md` plus official sites — never paths inside `node_modules/`.

## Agent execution environment

Commands from `shell_run` and `run_tests` execute in the **tdd-agentic backend container**
(`python:3.12-slim` + Node.js LTS), mounted at `/app/workspace/<project_id>/`.

| Property | Typical value |
|----------|----------------|
| OS | Debian (from `python:3.12-slim`; exact release varies) |
| Architectures | `aarch64` (Apple Silicon / ARM servers) or `x86_64` |
| Node / npm | Installed globally in the backend image |
| Docker CLI | **Not** available inside the agent container |
| `mongod` | Not pre-installed unless added to the backend image |

Your system prompt each turn also includes **detected** OS/arch for this deployment.

### Native binaries vs npm

- `npm install` only adds JavaScript; packages like **mongodb-memory-server** and **Playwright**
  still download large **platform-specific binaries**. A `403` from `fastdl.mongodb.org` usually
  means that **distro/arch/version tarball does not exist** — not a flaky network.
- On **Linux arm64 + Debian**, MongoDB does not publish official arm64 Debian builds. Use
  `MONGOMS_DISTRO=ubuntu-22.04` (and MongoDB `>=7.0.3` on recent Debian), or avoid
  `mongodb-memory-server` in agent runs (injected mongoose / real `MONGODB_URI`).

### What not to do

- Do not assume project `docker-compose.yml` services are reachable from agent commands unless
  documented (agents are not on the project compose network by default).
- Do not run `docker compose` from agents (no Docker socket in the backend container).
- After the same binary-download error **3+** times, mark the subtask **blocked** and escalate.
