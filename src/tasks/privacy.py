"""
Resume privacy wipe task.

Scheduled via Celery apply_async(countdown=900) at the end of each analysis task.
Wipes parsed_text and replaces sections JSONB with only the pre-computed metadata
(inferences, evidence) so the results page continues to work post-wipe.

Raw file bytes are never stored on disk — they exist only in-memory during parsing
and in the Celery task message (ephemeral Redis). Only parsed_text and sections
are persisted to the DB and are subject to this wipe.
"""

import uuid

import structlog

from src.database import get_db
from src.models.resume import Resume
from src.worker import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.wipe_resume_text",
    ignore_result=True,
    max_retries=3,
    default_retry_delay=60,
)
def wipe_resume_text(self, session_id: str) -> None:
    """
    Wipe PII resume text 15 minutes after analysis completes.

    Preserves pre-computed _inferences and _evidence metadata in sections JSONB
    so the results page evidence panel remains functional after the wipe.
    Status is set to "wiped" so subsequent calls are idempotent.
    """
    try:
        with get_db() as db:
            resume = db.query(Resume).filter(
                Resume.session_id == uuid.UUID(session_id)
            ).first()

            if not resume:
                log.warning("wipe_resume_not_found", session_id=session_id)
                return

            if resume.status == "wiped":
                return  # already wiped — idempotent

            # Preserve pre-computed metadata embedded during the analysis task
            retained_meta: dict = {}
            if isinstance(resume.sections, dict):
                retained_meta["_inferences"] = resume.sections.get("_inferences", [])
                retained_meta["_evidence"] = resume.sections.get("_evidence", {})
            retained_meta["_wiped"] = True

            resume.parsed_text = None
            resume.sections = retained_meta
            resume.status = "wiped"

        log.info("resume_text_wiped", session_id=session_id)

    except Exception as exc:
        log.error("wipe_resume_error", session_id=session_id, error=str(exc))
        raise self.retry(exc=exc)  # type: ignore[name-defined]
