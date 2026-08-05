"""
Task status polling endpoint.

GET /v1/tasks/{session_id}
  - Returns current task status, progress, and result when complete.
  - Frontend polls this every 2s while on the /processing page.
"""

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database import get_db_session as get_db
from src.models.task import Task

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


@router.get("/{session_id}")
async def get_task_status(session_id: str, db: Session = Depends(get_db)):
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

    task = db.query(Task).filter(Task.session_id == sid).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    response = {
        "session_id": session_id,
        "status": task.status,
        "progress_pct": task.progress_pct,
        "progress_message": task.progress_message,
        "task_type": task.task_type,
        "queued_at": task.queued_at.isoformat() if task.queued_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }

    if task.status == "complete":
        response["result"] = task.result
        response["results_url"] = f"/results/{session_id}"

    if task.status == "failed":
        response["error"] = task.error

    return response
