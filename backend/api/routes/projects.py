"""Project CRUD routes."""
from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import AsyncQdrantClient
from sqlalchemy import delete
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.events import Event, bus
from backend.api.routes.agents import RUNNING_TASKS
from backend.config import get_settings
from backend.db.session import get_db
from backend.rag.ingestion import collection_name
from backend.ticket_system import schemas, service
from backend.ticket_system.models import AgentLog

router = APIRouter()


@router.post("", response_model=schemas.ProjectOut)
async def create(payload: schemas.ProjectCreate, db: AsyncSession = Depends(get_db)):
    return await service.create_project(db, payload)


@router.get("", response_model=list[schemas.ProjectOut])
async def list_all(db: AsyncSession = Depends(get_db)):
    return await service.list_projects(db)


@router.get("/{project_id}", response_model=schemas.ProjectOut)
async def get_one(project_id: str, db: AsyncSession = Depends(get_db)):
    try:
        return await service.get_project(db, project_id)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{project_id}")
async def delete_one(project_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Delete a project and every artifact attached to it.

    Cascades:
    - Cancels any in-flight graph run for the project.
    - Removes the project row (CASCADE drops tickets, subtasks, todos).
    - Drops the Qdrant collection holding the project's RAG vectors.
    - Wipes the project's workspace directory.

    LangGraph checkpoints in PostgreSQL are left in place (they're keyed by
    ``thread_id`` and harmless once the project is gone). If you want to
    reclaim that space, drop the ``checkpoint_blobs``/``checkpoints`` rows
    matching ``thread_id = <project_id>`` manually.
    """
    try:
        project = await service.get_project(db, project_id)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    task = RUNNING_TASKS.get(project_id)
    if task is not None and not task.done():
        task.cancel()

    settings = get_settings()
    try:
        client = AsyncQdrantClient(url=settings.qdrant_url)
        try:
            await client.delete_collection(collection_name=collection_name(project_id))
        finally:
            await client.close()
    except Exception:
        pass  # collection may not exist

    workspace = settings.workspace_root / project_id
    if workspace.exists():
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            pass

    await db.execute(delete(AgentLog).where(AgentLog.project_id == project_id))
    await db.delete(project)
    await db.commit()

    await bus.publish(
        Event(type="project", payload={"action": "deleted", "id": project_id})
    )
    return {"status": "deleted", "project_id": project_id}
