# TDD Agentic Development System

A fully orchestrated, **TDD-first** agentic software development system built on **LangGraph** + **LangChain**.

A team of specialized AI agents — Project Manager, Researcher, Lead, Coordinator, Backend Dev, Frontend Dev, DevOps, and QA — collaborate to deliver software using strict Test-Driven Development. The Lead plans all subtasks (backend + frontend) as structured JSON; the Coordinator persists plans to the ticket DB. A simplified ticket platform tracks work end-to-end, a CRAG-style RAG system gives every agent project-specific context, and a real-time monitoring client lets you watch, interrupt, and steer the run.

---

## Architecture

```
┌────────────────────────────┐      ┌─────────────────────────────┐
│   Next.js Monitoring UI    │◀────▶│  FastAPI + WebSocket Hub    │
│  (Tickets, Logs, HITL)     │      │                             │
└────────────────────────────┘      └─────────────┬───────────────┘
                                                  │
                                                  ▼
                                ┌────────────────────────────────────┐
                                │  LangGraph Root Graph (Supervisor) │
                                │                                    │
                                │   Project Manager                  │
                                │     │                              │
                                │     ├─ researcher   (subgraph)     │
                                │     ├─ lead         (plan only)    │
                                │     ├─ coordinator  (persist plan) │
                                │     ├─ backend_dev, frontend_dev   │
                                │     └─ devops, qa                  │
                                └────────────────────────────────────┘
                                                  │
                ┌─────────────────────────────────┼────────────────────────┐
                ▼                                 ▼                        ▼
        PostgreSQL                            Qdrant                  Langfuse
   (tickets + checkpointer)             (per-project RAG)          (LLM tracing)
```

### Orchestration flow

Every specialist returns to the PM. Typical greenfield path:

1. **Researcher** — scaffold docs, ingest into RAG, optionally create skills.
2. **PM** — creates tickets, routes using structured `RoutingDecision` output (validated against the DB).
3. **Lead** — cognitive-only (no tools); outputs `execution_plan` JSON with RITE test cases for backend, frontend, devops, and QA subtasks.
4. **Coordinator** — reads `state.execution_plan`, persists subtasks via `save_execution_plan()`.
5. **Devs / DevOps / QA** — phased toolsets: ticket tools first, then code/fs/shell tools after engaging a subtask.

Handoffs use a compact `[from agent → agent]` protocol with optional `context_refs` into a cross-run context store. See `docs/tool-calling-and-prompts-analysis.md` for prompt/tool design details.

### Agent hierarchy

| Agent | Role | Tools (summary) |
|-------|------|-----------------|
| Project Manager | Supervisor; routes work, manages tickets, asks humans | Ticket tools + RAG + HITL |
| Researcher | Web search, doc generation, RAG ingestion, skill creation | Search, RAG, fs, skills |
| Lead | Plans all subtasks (backend + frontend) in one `execution_plan` | None (JSON output only) |
| Coordinator | Persists Lead plans to the ticket DB | Persistence + RAG |
| Backend / Frontend Dev | TDD red→green→refactor on assigned subtasks | Phased: ticket → code |
| DevOps | Infra, CI, Docker subtasks | Phased: ticket → code |
| QA | Integration / e2e tests | Phased: ticket → code |

### Ticket lifecycle

```
Draft → InReview → QuestionsPending → InReview → Todo → InProgress → Done
```

Tickets contain `business_requirements`, `technical_requirements`, and an ordered list of subtasks. Subtasks contain `test_cases` (the TDD anchor) and an ordered list of todos.

---

## Quick start

### 1. Configure

```bash
cp .env.example .env
# fill in OPENAI_API_KEY (or only ANTHROPIC_API_KEY if you don't use OpenAI models)
```

### 2. Boot the stack

```bash
docker compose up --build
```

Services:

- **Backend** — http://localhost:8000 (FastAPI + WebSocket at `/ws`)
- **Frontend** — http://localhost:3000 (Next.js monitoring client)
- **PostgreSQL** — `localhost:5432` (tickets + LangGraph checkpointer)
- **Qdrant** — http://localhost:6333 (per-project vector collections)
- **Langfuse** — http://localhost:3001 (optional LLM tracing)

For dev mode with hot reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### 3. Use it

Open http://localhost:3000:

1. **Dashboard** — create a project, click *Start agent run*. Pending interrupts and the live agent feed appear here.
2. **Tickets** — Kanban board across the full lifecycle. Click any ticket to inspect its requirements, subtasks, test cases, and todos.
3. **Logs** — full-screen real-time agent activity feed.

When the PM raises a question (via `ask_human`), an interrupt card appears on the dashboard. Type your answer and click *Send & resume* to continue the graph.

---

## Project layout

```
tdd-agentic/
├── backend/
│   ├── agents/                # LangGraph nodes & subgraphs
│   │   ├── graph.py           # Root graph wiring
│   │   ├── state.py           # SystemState, ExecutionPlan, SubtaskPlan
│   │   ├── runner.py          # Specialist subgraph factory (phased tools, plan parsing)
│   │   ├── handoff.py         # Compact inter-agent handoff protocol
│   │   ├── context_store.py   # Cross-agent context refs (ctx_N pointers)
│   │   ├── checkpointer.py    # AsyncPostgresSaver
│   │   ├── llm.py             # Multi-provider LLM factory
│   │   ├── observability.py   # Optional Langfuse hook
│   │   ├── prompts.py         # System prompts per role
│   │   ├── project_manager/   # Supervisor (structured routing)
│   │   ├── researcher/
│   │   ├── leads/lead/        # Merged Lead (backend + frontend planning)
│   │   ├── coordinator/       # DB persistence for execution plans
│   │   ├── developers/        # backend_dev, frontend_dev, devops, qa
│   │   └── skills/            # registry.py, loader.py
│   ├── ticket_system/         # SQLAlchemy models + state machine service
│   ├── rag/                   # Qdrant ingestion + CRAG retrieval
│   ├── tools/                 # LangChain tools (ticket, persistence, RAG, code, web, HITL)
│   ├── api/                   # FastAPI routes + WebSocket hub
│   └── db/                    # Async engine + Alembic migrations
├── docs/                      # Architecture & prompt/tool analysis
└── frontend/                  # Next.js 15 + Tailwind monitoring client
    └── src/
        ├── app/               # Dashboard, tickets, logs pages
        ├── components/        # Tickets board, agent log, HITL panel
        └── lib/               # API client, WebSocket hook, Zustand store
```

---

## How TDD is enforced

1. **Lead outputs RITE `test_cases` in `execution_plan`** — each subtask (except devops/qa) must include given/should/expected specs. The Coordinator persists them to the ticket DB.
2. **Devs follow red → green → refactor**: write the failing tests first via `fs_write`, run them via `run_tests` (must fail), then implement code to make them pass.
3. **Subtasks are strictly ordered.** A dev calls `next_pending_subtask_in_project` (pass `ticket_id` when PM routed to a specific ticket) and completes subtasks in lead-defined order.
4. **Verification gates.** DevOps and similar roles must pass `run_tests` before marking a subtask done; incomplete turns are flagged for PM re-dispatch.
5. **Done = all subtasks done.** The PM only marks a ticket DONE after every subtask is DONE.

---

## Multi-provider LLM

All agent LLM choices come from `.env`:

```
PM_MODEL=anthropic/claude-sonnet-4-6
RESEARCHER_MODEL=openai/gpt-4o
LEAD_MODEL=anthropic/claude-sonnet-4-6
DEV_MODEL=anthropic/claude-sonnet-4-6
COORDINATOR_MODEL=anthropic/claude-sonnet-4-6   # optional; defaults to DEV_MODEL
BACKEND_DEV_MODEL=...                            # optional per-role overrides
FRONTEND_DEV_MODEL=...
DEVOPS_MODEL=...
QA_MODEL=...
GRADER_MODEL=anthropic/claude-haiku-4-5
DEV_AGENT_MAX_STEPS=60                           # LLM↔tool loop budget per dev turn
```

Slugs are `provider/model`. Supported providers: `openai`, `anthropic`, and `openrouter` (OpenAI-compatible gateway — set `OPENROUTER_API_KEY` and use OpenRouter model ids, e.g. `openrouter/anthropic/claude-sonnet-4`).

For local inference (llama.cpp, LM Studio, vLLM), set `OPENAI_BASE_URL` and use `openai/<model-id>` slugs. The PM uses structured output for routing; smaller local models may need a cloud PM while other roles stay local — see `.env.example` and `docs/local-model-finetuning-research.md`.

---

## RAG — CRAG pipeline

`backend/rag/retrieval.py` implements **Corrective RAG**:

1. Vector retrieve from the project's Qdrant collection.
2. Grade each doc with a small LLM (`GRADER_MODEL`) for relevance.
3. If nothing relevant survives, **rewrite** the query and retry once.

Embeddings default to OpenAI `text-embedding-3-small`. Set `EMBEDDING_PROVIDER=local` to use sentence-transformers locally (optional install — see below).

---

## Skills system

Researchers can call the `create_skill` tool to author a focused skill brief and assign it to one or more agent roles. The brief is:

- Persisted as `_skills/<name>/SKILL.md` in the workspace
- Registered in `_skills/registry.json`
- Indexed into the project's RAG collection
- **Automatically injected** into the system prompt of every agent assigned to that role (`backend/agents/skills/loader.py`)

---

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/tool-calling-and-prompts-analysis.md`](docs/tool-calling-and-prompts-analysis.md) | Prompt/tool design, structured routing, phased toolsets, implementation status |
| [`docs/local-model-finetuning-research.md`](docs/local-model-finetuning-research.md) | Per-role local model strategy and fine-tuning datasets |
| [`docs/agent-efficiency-plan.md`](docs/agent-efficiency-plan.md) | Token efficiency patterns (handoffs, context store, compact ticket reads) |

---

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/projects` | Create a project |
| `GET`  | `/api/projects` | List projects |
| `POST` | `/api/tickets` | Create a ticket |
| `GET`  | `/api/tickets?project_id=...` | List tickets |
| `GET`  | `/api/tickets/{id}` | Get a ticket |
| `PATCH`| `/api/tickets/{id}` | Update / transition a ticket |
| `POST` | `/api/tickets/{id}/answer` | Answer a pending question |
| `POST` | `/api/agents/start` | Start an agent run for a project |
| `POST` | `/api/agents/resume` | Resume an interrupted run |
| `GET`  | `/api/agents/state/{project_id}` | Inspect graph state |
| `GET`  | `/api/agents/interrupts/{project_id}` | List pending interrupts |
| `WS`   | `/ws` | Real-time event stream |

---

## Local development without Docker

Core backend (OpenAI embeddings, no PyTorch):

```bash
python scripts/install_backend.py --dev
```

Local embeddings (`EMBEDDING_PROVIDER=local`) — PyTorch is chosen from your hardware (`auto` = NVIDIA GPU → CUDA wheels, otherwise CPU on Linux/Windows, PyPI on macOS):

```bash
python scripts/install_backend.py --local-embeddings --dev
# Force CPU everywhere:  python scripts/install_backend.py --local-embeddings --torch cpu
# Force CUDA (NVIDIA):   TORCH_DEVICE=cuda python scripts/install_backend.py --local-embeddings
```

Then start services (create the DB and Qdrant manually):

```bash
uvicorn backend.api.main:app --reload
cd frontend && npm install && npm run dev
```

Docker with local embeddings (two options):

**A. Official PyTorch image as backend base** (recommended in Docker — torch is pre-installed, no pip CUDA download):

```bash
# Set EMBEDDING_PROVIDER=local in .env first
docker compose -f docker-compose.yml -f docker-compose.pytorch.yml up --build
# NVIDIA GPU in container:
docker compose -f docker-compose.yml -f docker-compose.pytorch.yml -f docker-compose.pytorch-gpu.yml up --build
```

**B. Slim Python image + pip-installed CPU/CUDA torch:**

```bash
INSTALL_LOCAL_EMBEDDINGS=true TORCH_DEVICE=cpu docker compose build backend
```

A separate `pytorch` service in Compose cannot share Python imports with the backend — PyTorch must live in the **same container** as uvicorn (or you run a dedicated embedding HTTP server and point the backend at it).

Run migrations:

```bash
alembic -c backend/db/alembic.ini upgrade head
# or for schema-from-models in dev:
python -c "import asyncio; from backend.db.session import init_db; asyncio.run(init_db())"
```
