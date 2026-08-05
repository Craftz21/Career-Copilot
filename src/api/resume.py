"""
Resume upload endpoint.

POST /v1/resume/upload
  - Accepts multipart form: file + target_role + duration
  - Validates file type, size, and duration
  - Enforces per-IP upload rate limit
  - Creates UserSession + Task records
  - Enqueues analyze_resume_task (Celery)
  - Returns 202 with session_id for polling
"""

import datetime
import threading
import time
import uuid
from collections import defaultdict
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.config import get_settings
from src.database import get_db_session as get_db
from src.models.resume import Resume
from src.models.session import UserSession
from src.models.task import Task
from src.tasks.analyze_resume import analyze_resume_task

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1/resume", tags=["resume"])
settings = get_settings()

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}

# Duration values must match the <select> options in index.html exactly.
_VALID_DURATIONS = {"4 weeks", "2 months", "3 months", "6 months", "12 months"}

# In-process upload rate limiter. Resets on worker restart (acceptable for
# a single free-tier Render worker). Keyed by client IP.
_rl_lock = threading.Lock()
_rl_timestamps: dict[str, list[float]] = defaultdict(list)


def _enforce_upload_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - 3600.0
    with _rl_lock:
        window = [t for t in _rl_timestamps[ip] if t > cutoff]
        if len(window) >= settings.rate_limit_upload_per_hour:
            raise HTTPException(
                status_code=429,
                detail="Too many uploads from this address. Try again in an hour.",
            )
        window.append(now)
        _rl_timestamps[ip] = window


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_resume(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF or DOCX resume")],
    target_role: Annotated[str, Form(description="Target job role, e.g. 'Backend Software Engineer'")],
    duration: Annotated[str, Form(description="Study duration")] = "3 months",
    db: Session = Depends(get_db),
):
    # --- Rate limit ---
    _enforce_upload_rate_limit(request)

    # --- Validate file type ---
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type. Upload a PDF or DOCX file.",
        )

    # --- Validate file size ---
    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb} MB.",
        )
    if len(file_bytes) < 100:
        raise HTTPException(status_code=422, detail="File appears to be empty.")

    # --- Validate role input ---
    target_role = target_role.strip()
    if not target_role or len(target_role) < 2:
        raise HTTPException(status_code=422, detail="Please enter a target role.")
    if len(target_role) > 200:
        raise HTTPException(status_code=422, detail="Target role name is too long.")

    # --- Validate duration ---
    duration = duration.strip()
    if duration not in _VALID_DURATIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid duration. Choose from: {', '.join(sorted(_VALID_DURATIONS))}.",
        )

    # --- Create session ---
    session_id = uuid.uuid4()
    expires_at = (
        datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        + datetime.timedelta(days=settings.session_ttl_days)
    )
    session = UserSession(
        session_id=session_id,
        target_role=target_role,
        status="queued",
        expires_at=expires_at,
    )
    db.add(session)

    # --- Create task record (for polling) ---
    task_record = Task(
        session_id=session_id,
        task_type="analyze_resume",
        status="queued",
        progress_pct=0,
        progress_message="Queued for processing…",
    )
    db.add(task_record)

    # --- Create resume stub ---
    safe_filename = file.filename or f"resume{suffix}"
    resume = Resume(
        session_id=session_id,
        filename=safe_filename,
        file_size_bytes=len(file_bytes),
        layout_type=suffix.lstrip("."),
        status="pending",
    )
    db.add(resume)
    db.flush()

    # --- Enqueue Celery task ---
    celery_task = analyze_resume_task.apply_async(
        kwargs={
            "session_id": str(session_id),
            "file_bytes": file_bytes,
            "filename": safe_filename,
            "target_role": target_role,
            "duration": duration,
        },
        task_id=str(uuid.uuid4()),
    )

    task_record.celery_task_id = celery_task.id

    log.info(
        "resume_upload_accepted",
        session_id=str(session_id),
        filename=safe_filename,
        file_size_bytes=len(file_bytes),
        target_role=target_role,
        celery_task_id=celery_task.id,
    )

    return JSONResponse(
        status_code=202,
        content={
            "session_id": str(session_id),
            "status": "queued",
            "poll_url": f"/v1/tasks/{session_id}",
            "processing_url": f"/processing/{session_id}",
        },
    )
