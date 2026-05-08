"""FastAPI application entrypoint.

Wires HTTP routes, the WebSocket hub, CORS, DB initialization, and the
LangGraph checkpointer pool lifecycle.
"""
from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager

# LangChain emits a pending-deprecation warning about the future default
# of ``allowed_objects`` from inside langgraph's encrypted-serializer
# import path. We don't use encrypted serialization, the warning is purely
# advisory, and there's nothing actionable on our side until the upstream
# default changes — silence just this one pattern so logs stay readable.
try:  # langchain-core may not always expose this class
    from langchain_core._api.deprecation import (  # type: ignore[attr-defined]
        LangChainPendingDeprecationWarning,
    )

    warnings.filterwarnings(
        "ignore",
        message=r"The default value of `allowed_objects` will change.*",
        category=LangChainPendingDeprecationWarning,
    )
except Exception:  # pragma: no cover - best-effort suppression
    warnings.filterwarnings(
        "ignore",
        message=r"The default value of `allowed_objects` will change.*",
    )

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.agents.checkpointer import close_pool, get_pool
from backend.agents.llm import log_resolved_llm_routing
from backend.agents.skills.seed import seed_builtin_skills
from backend.api.routes import agents as agents_routes
from backend.api.routes import projects as projects_routes
from backend.api.routes import tickets as tickets_routes
from backend.api.routes.agents import cancel_all_running_tasks
from backend.api.websocket import router as ws_router
from backend.config import get_settings
from backend.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log_resolved_llm_routing()
    ll = logging.getLogger("backend.agents.llm_http")
    ll.setLevel(logging.INFO if settings.llm_invoke_log_each_call else logging.WARNING)
    if settings.llm_openai_http_debug:
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
    await init_db()
    await get_pool()  # warm checkpointer connection pool
    await seed_builtin_skills()
    yield
    # Shutdown: cancel in-flight agent runs FIRST so uvicorn's drain
    # phase doesn't sit forever waiting on orphan LLM calls / retries.
    await cancel_all_running_tasks()
    await close_pool()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="TDD Agentic Dev System",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ws_router)
    app.include_router(projects_routes.router, prefix="/api/projects", tags=["projects"])
    app.include_router(tickets_routes.router, prefix="/api/tickets", tags=["tickets"])
    app.include_router(agents_routes.router, prefix="/api/agents", tags=["agents"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
