# TDD Agentic Development System

A fully orchestrated, **TDD-first** agentic software development system built on **LangGraph** + **LangChain**.

A team of specialized AI agents — Project Manager, Researcher, Backend Lead, Frontend Lead, Backend Dev, Frontend Dev, DevOps, and QA — collaborate to deliver software using strict Test-Driven Development. A simplified ticket platform tracks work end-to-end, a CRAG-style RAG system gives every agent project-specific context, and a real-time monitoring client lets you watch, interrupt, and steer the run.

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
                                │     ├─ researcher  (subgraph)      │
                                │     ├─ backend_lead, frontend_lead │
                                │     ├─ backend_dev, frontend_dev   │
                                │     └─ devops, qa                  │
                                └────────────────────────────────────┘
                                                  │
                ┌─────────────────────────────────┼────────────────────────┐
                ▼                                 ▼                        ▼
        PostgreSQL                            Qdrant                  Langfuse
   (tickets + checkpointer)             (per-project RAG)          (LLM tracing)
```

### Agent hierarchy

| Agent | Role |
|-------|------|
| Project Manager | Supervisor; generates tickets, routes work, asks humans questions |
| Researcher | Web search, doc generation, RAG ingestion, skill creation |
| Backend / Frontend Leads | Decompose tickets into ordered subtasks with explicit test cases |
| Backend / Frontend Devs | TDD red→green→refactor cycle on assigned subtasks |
| DevOps | Infra, CI, Docker subtasks |
| QA | Integration / e2e tests |

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

When the PM (or a Lead) raises a question, an interrupt card appears on the dashboard. Type your answer and click *Send & resume* to continue the graph.

---

## Project layout

```
tdd-agentic/
├── backend/
│   ├── agents/                # LangGraph nodes & subgraphs
│   │   ├── graph.py           # Root graph wiring
│   │   ├── state.py           # Pydantic SystemState (extra='forbid')
│   │   ├── runner.py          # Generic specialist subgraph factory
│   │   ├── checkpointer.py    # AsyncPostgresSaver
│   │   ├── llm.py             # Multi-provider LLM factory
│   │   ├── observability.py   # Optional Langfuse hook
│   │   ├── prompts.py         # System prompts per role
│   │   ├── project_manager/   # Supervisor
│   │   ├── researcher/
│   │   ├── leads/             # backend_lead, frontend_lead
│   │   ├── developers/        # backend_dev, frontend_dev, devops, qa
│   │   └── skills/            # registry.py, loader.py
│   ├── ticket_system/         # SQLAlchemy models + state machine service
│   ├── rag/                   # Qdrant ingestion + CRAG retrieval
│   ├── tools/                 # LangChain tools (ticket, RAG, code, web, HITL)
│   ├── api/                   # FastAPI routes + WebSocket hub
│   └── db/                    # Async engine + Alembic migrations
└── frontend/                  # Next.js 15 + Tailwind monitoring client
    └── src/
        ├── app/               # Dashboard, tickets, logs pages
        ├── components/        # Tickets board, agent log, HITL panel
        └── lib/               # API client, WebSocket hook, Zustand store
```

---

## How TDD is enforced

1. **Leads must produce explicit `test_cases`** when creating a subtask — without test cases, the subtask is incomplete.
2. **Devs follow red → green → refactor**: write the failing tests first via `fs_write`, run them via `run_tests` (must fail), then implement code to make them pass.
3. **Subtasks are strictly ordered.** A dev must always pull `next_pending_subtask` and complete subtasks in the order the lead set.
4. **Done = all subtasks done.** The PM only marks a ticket DONE after every subtask is DONE.

---

## Multi-provider LLM

All agent LLM choices come from `.env`:

```
PM_MODEL=anthropic/claude-sonnet-4-6
RESEARCHER_MODEL=openai/gpt-4o
LEAD_MODEL=anthropic/claude-sonnet-4-6
DEV_MODEL=anthropic/claude-sonnet-4-6
GRADER_MODEL=anthropic/claude-haiku-4-5
```

Slugs are `provider/model`. Add new providers by extending `backend/agents/llm.py`.

---

## RAG — CRAG pipeline

`backend/rag/retrieval.py` implements **Corrective RAG**:

1. Vector retrieve from the project's Qdrant collection.
2. Grade each doc with a small LLM (`GRADER_MODEL`) for relevance.
3. If nothing relevant survives, **rewrite** the query and retry once.

Embeddings default to OpenAI `text-embedding-3-small`. Set `EMBEDDING_PROVIDER=local` to use sentence-transformers locally.

---

## Skills system

Researchers can call the `create_skill` tool to author a focused skill brief and assign it to one or more agent roles. The brief is:

- Persisted as `_skills/<name>/SKILL.md` in the workspace
- Registered in `_skills/registry.json`
- Indexed into the project's RAG collection
- **Automatically injected** into the system prompt of every agent assigned to that role (`backend/agents/skills/loader.py`)

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

```bash
pip install -e .[dev]
# create the DB and Qdrant manually, then:
uvicorn backend.api.main:app --reload
cd frontend && npm install && npm run dev
```

Run migrations:

```bash
alembic -c backend/db/alembic.ini upgrade head
# or for schema-from-models in dev:
python -c "import asyncio; from backend.db.session import init_db; asyncio.run(init_db())"
```
