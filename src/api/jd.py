"""
JD Match endpoint.

POST /v1/jd/analyze
  - Accepts multipart form: file (resume) + jd_text + job_title + company_name + duration
  - Reuses all existing validation from the resume upload endpoint
  - Creates UserSession + Task + Resume + JDAnalysis records
  - Enqueues analyze_jd_task (Celery)
  - Returns 202 with session_id — same polling/results flow as standard analysis
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
from src.models.jd import JDAnalysis
from src.models.resume import Resume
from src.models.session import UserSession
from src.models.task import Task
from src.tasks.analyze_jd import analyze_jd_task

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1/jd", tags=["jd"])
settings = get_settings()

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
_VALID_DURATIONS = {"4 weeks", "2 months", "3 months", "6 months", "12 months"}
_JD_MIN_CHARS = 100
_JD_MAX_CHARS = 15_000

# Shared rate limit with resume uploads (same IP bucket, same hourly window).
_rl_lock = threading.Lock()
_rl_timestamps: dict[str, list[float]] = defaultdict(list)


def _enforce_rate_limit(request: Request) -> None:
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


@router.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
async def analyze_jd(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF or DOCX resume")],
    jd_text: Annotated[str, Form(description="Raw job description text")],
    job_title: Annotated[str, Form(description="Job title from the JD")] = "",
    company_name: Annotated[str, Form(description="Company name (optional)")] = "",
    duration: Annotated[str, Form(description="Study duration")] = "3 months",
    db: Session = Depends(get_db),
):
    _enforce_rate_limit(request)

    # --- Validate resume file ---
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Unsupported file type. Upload a PDF or DOCX file.")

    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {settings.max_upload_size_mb} MB.")
    if len(file_bytes) < 100:
        raise HTTPException(status_code=422, detail="File appears to be empty.")

    # --- Validate JD text ---
    jd_text = jd_text.strip()
    if len(jd_text) < _JD_MIN_CHARS:
        raise HTTPException(status_code=422, detail=f"Job description is too short. Paste the full JD (at least {_JD_MIN_CHARS} characters).")
    if len(jd_text) > _JD_MAX_CHARS:
        raise HTTPException(status_code=422, detail="Job description is too long. Please trim it to the key requirements section.")

    # --- Validate optional fields ---
    job_title = job_title.strip()[:200]
    company_name = company_name.strip()[:200]

    duration = duration.strip()
    if duration not in _VALID_DURATIONS:
        raise HTTPException(status_code=422, detail=f"Invalid duration. Choose from: {', '.join(sorted(_VALID_DURATIONS))}.")

    # --- Create session ---
    session_id = uuid.uuid4()
    # target_role stores the job title for display in the processing/results pages.
    # Falls back to "the target role" if job_title is blank.
    display_role = job_title or "Target Role from JD"
    expires_at = (
        datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        + datetime.timedelta(days=settings.session_ttl_days)
    )
    session = UserSession(
        session_id=session_id,
        target_role=display_role,
        status="queued",
        expires_at=expires_at,
    )
    db.add(session)

    # --- Create task record ---
    task_record = Task(
        session_id=session_id,
        task_type="analyze_jd",
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

    # --- Create JDAnalysis stub (jd_skills populated by worker) ---
    jd_record = JDAnalysis(
        session_id=session_id,
        jd_raw_text=jd_text,
        company_name=company_name or None,
        job_title=job_title or None,
    )
    db.add(jd_record)
    db.flush()

    # --- Enqueue Celery task ---
    celery_task = analyze_jd_task.apply_async(
        kwargs={
            "session_id": str(session_id),
            "file_bytes": file_bytes,
            "filename": safe_filename,
            "jd_text": jd_text,
            "job_title": job_title,
            "company_name": company_name,
            "duration": duration,
        },
        task_id=str(uuid.uuid4()),
    )

    task_record.celery_task_id = celery_task.id

    log.info(
        "jd_analyze_accepted",
        session_id=str(session_id),
        filename=safe_filename,
        jd_chars=len(jd_text),
        job_title=job_title,
        company_name=company_name,
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
