"""Async SQLAlchemy engine, session factory, and DeclarativeBase."""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_settings


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a session and ensures cleanup."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables (used in dev / CI; production should use Alembic)."""
    from backend.ticket_system import models  # noqa: F401  (import for metadata side-effect)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Hot-path indexes for log/history views. `create_all` doesn't add new
        # indexes to existing tables; this keeps local dev DBs fast without a
        # full migration workflow.
        await conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS agent_logs_project_created_at_desc_idx
                ON agent_logs (project_id, created_at DESC);
                """
            )
        )
