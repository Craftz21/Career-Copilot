"""
Celery task: full resume analysis pipeline.

Steps executed by the worker:
  1. Update task status → "processing" (10 %)
  2. Parse resume bytes → raw_text + sections            [Phase 1 DB context]
  3. Normalize target role → role_id                    [Phase 1 DB context]
  4. Extract skills (two-pass)                          [Phase 1 DB context]
  5. Persist user_skills to DB                          [Phase 1 DB context]
  6. Run gap analysis                                   [Phase 1 DB context]
     → Phase 1 DB connection released before LLM call
  7. Update task status → "processing" (75 %)
  8. Generate learning roadmap (LLM)                    [no DB connection held]
  9. Update task status → "processing" (95 %)           [Phase 2 DB context]
 10. Persist roadmap + readiness_score to DB            [Phase 2 DB context]
 11. Update task status → "complete" (100 %)

Performance optimisations applied:
  O4 — _update_task calls reduced from 7 → 4 (saves 3 redundant DB connections).
  O6 — DB transaction split into two short-lived phases. The 4–15 s Groq LLM
       call (step 8) runs outside any DB context, so no idle connection is held
       during inference.

On any unhandled exception:
  - Update task status → "failed" with error message.
  - task_acks_late=True means the task re-queues on worker crash before step 10.
"""

import datetime
import uuid
from typing import Any

import structlog

from src.database import get_db
from src.models.resume import Resume
from src.models.roadmap import Roadmap
from src.models.session import UserSession
from src.models.task import Task
from src.models.user_skill import UserSkill
from src.services.gap_analyzer import analyze_gap
from src.services.resume_parser import ParseError, parse_resume
from src.services.roadmap_generator import generate_roadmap
from src.services.role_normalizer import RoleMatch, normalize_role
from src.services.skill_extractor import extract_skills
from src.worker import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.analyze_resume",
    max_retries=2,
    default_retry_delay=10,
    time_limit=300,       # hard kill after 5 minutes
    soft_time_limit=270,  # raises SoftTimeLimitExceeded at 4.5 min for cleanup
)
def analyze_resume_task(
    self,
    session_id: str,
    file_bytes: bytes,
    filename: str,
    target_role: str,
    duration: str = "3 months",
) -> dict[str, Any]:
    """
    Full pipeline task. Returns the complete result payload on success.
    Errors are stored in the tasks table — never raised to the caller.
    """
    # ── O4: 10 % — parsing started ───────────────────────────────────────────
    _update_task(session_id, status="processing", pct=10, msg="Parsing resume…")

    try:
        log.info("analyze_resume_start", session_id=session_id, filename=filename)

        # ── Phase 1: fast DB work (no LLM, all I/O is local or low-latency) ──
        with get_db() as db:
            try:
                parsed = parse_resume(file_bytes, filename)
            except ParseError as exc:
                return _fail(session_id, str(exc))

            # Persist parsed resume
            resume = db.query(Resume).filter(Resume.session_id == uuid.UUID(session_id)).first()
            if resume:
                resume.parsed_text = parsed["raw_text"]
                resume.sections = parsed["sections"]
                resume.layout_type = parsed["layout_type"]
                resume.status = "parsed"
            else:
                resume = Resume(
                    session_id=uuid.UUID(session_id),
                    layout_type=parsed["layout_type"],
                    status="parsed",
                    parsed_text=parsed["raw_text"],
                    sections=parsed["sections"],
                )
                db.add(resume)
            db.flush()

            # Normalize role
            role_match = normalize_role(target_role, db)

            if role_match.match_type == "semantic_suggest":
                if role_match.confidence >= 0.70 and role_match.suggestions:
                    # High-confidence fuzzy suggestion: auto-select the best match.
                    # Happens for generic titles ("Software Engineer") that aren't
                    # explicit aliases but score well against a specific canonical role.
                    top = role_match.suggestions[0]
                    log.info(
                        "role_auto_selected",
                        input=target_role,
                        selected=top["display_name"],
                        confidence=round(role_match.confidence, 3),
                    )
                    role_match = RoleMatch(
                        role_id=top["role_id"],
                        canonical_name=None,
                        display_name=top["display_name"],
                        confidence=role_match.confidence,
                        match_type="fuzzy_match",
                        suggestions=[],
                    )
                else:
                    return _fail(
                        session_id,
                        f"Could not identify role '{target_role}'. "
                        f"Suggestions: {[s['display_name'] for s in role_match.suggestions[:3]]}",
                    )
            elif role_match.match_type == "no_match":
                return _fail(
                    session_id,
                    f"Could not identify role '{target_role}'. No close match found.",
                )

            role_id = role_match.role_id

            # Update session with resolved role_id
            session = db.query(UserSession).filter(
                UserSession.session_id == uuid.UUID(session_id)
            ).first()
            if session:
                session.role_id = role_id
                session.status = "processing"
            db.flush()

            # ── O4: 40 % — skill extraction started ──────────────────────────
            _update_task(session_id, status="processing", pct=40, msg="Extracting skills…")

            skill_results = extract_skills(parsed["sections"], parsed["raw_text"], db)

            # Persist user_skills (upsert: delete old, insert new)
            db.query(UserSkill).filter(
                UserSkill.session_id == uuid.UUID(session_id)
            ).delete()
            for s in skill_results:
                db.add(UserSkill(
                    session_id=uuid.UUID(session_id),
                    skill_id=s["skill_id"],
                    confidence=s["confidence"],
                    source=s["source"],
                ))
            db.flush()

            # ── O4: 65 % — gap analysis starting ──────────────────────────────
            _update_task(session_id, status="processing", pct=65, msg="Analyzing skill gaps…")
            gap_data = analyze_gap(session_id, role_id, db)

            # Phase 1 commits here and releases the connection before the LLM call

        # ── O4: 75 % — roadmap generation starting (outside DB context) ──────
        _update_task(session_id, status="processing", pct=75, msg="Generating roadmap…")

        # ── Phase 2: LLM call — no DB connection held during inference ────────
        # O6: generate_roadmap() manages its own short-lived DB connections
        roadmap_result = generate_roadmap(
            session_id=session_id,
            role_display_name=role_match.display_name or target_role,
            gap_data=gap_data,
            duration=duration,
        )

        # ── Phase 3: persist roadmap + finalize session ───────────────────────
        with get_db() as db:
            existing_roadmap = db.query(Roadmap).filter(
                Roadmap.session_id == uuid.UUID(session_id)
            ).first()
            if existing_roadmap:
                existing_roadmap.content = roadmap_result["content"]
                existing_roadmap.model_used = roadmap_result["model_used"]
                existing_roadmap.generation_ms = roadmap_result["generation_ms"]
                existing_roadmap.cache_hit = roadmap_result["cache_hit"]
                existing_roadmap.prompt_version = roadmap_result["prompt_version"]
            else:
                db.add(Roadmap(
                    session_id=uuid.UUID(session_id),
                    role_id=role_id,
                    duration=duration,
                    content=roadmap_result["content"],
                    model_used=roadmap_result["model_used"],
                    generation_ms=roadmap_result["generation_ms"],
                    cache_hit=roadmap_result["cache_hit"],
                    prompt_version=roadmap_result["prompt_version"],
                ))

            session = db.query(UserSession).filter(
                UserSession.session_id == uuid.UUID(session_id)
            ).first()
            if session:
                session.readiness_score = gap_data["readiness_score"]
                session.status = "complete"

            # ── F9 + F3 + F1: cache evidence/inferences in resume.sections ───
            # Computed here (while parsed sections are available) so the results
            # page can display evidence even after the privacy wipe removes the raw text.
            from src.services.soft_skill_inferencer import (
                build_skill_evidence,
                infer_soft_skills,
            )
            resume_record = db.query(Resume).filter(
                Resume.session_id == uuid.UUID(session_id)
            ).first()
            if resume_record and isinstance(resume_record.sections, dict):
                inferences = infer_soft_skills(resume_record.sections)
                evidence = build_skill_evidence(
                    resume_record.sections, gap_data.get("matched_skills", [])
                )
                # Embed metadata alongside existing sections (wipe task will
                # replace with metadata-only once the TTL expires).
                resume_record.sections = {
                    **resume_record.sections,
                    "_inferences": inferences,
                    "_evidence": {str(k): v for k, v in evidence.items()},
                }

            # Phase 3 commits here

        # ── F9: Schedule privacy wipe — 15 minutes TTL ───────────────────────
        from src.tasks.privacy import wipe_resume_text
        wipe_resume_text.apply_async(args=[session_id], countdown=900)

        result_payload = {
            "session_id": session_id,
            "readiness_score": gap_data["readiness_score"],
            "matched_skills_count": len(gap_data["matched_skills"]),
            "missing_skills_count": len(gap_data["missing_skills"]),
        }

        # ── O4: 100 % — done ─────────────────────────────────────────────────
        _update_task(
            session_id, status="complete", pct=100,
            msg="Analysis complete!", result=result_payload,
        )

        log.info(
            "analyze_resume_complete",
            session_id=session_id,
            readiness_score=gap_data["readiness_score"],
        )
        return result_payload

    except Exception as exc:
        log.exception("analyze_resume_unhandled_error", session_id=session_id)
        return _fail(session_id, f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_task(
    session_id: str,
    status: str,
    pct: int,
    msg: str,
    result: dict | None = None,
) -> None:
    """
    Open a dedicated short-lived DB connection to update task progress.
    Intentionally separate from the main pipeline context so that updates
    are immediately visible to polling clients (progress bar).
    """
    with get_db() as db:
        task = db.query(Task).filter(
            Task.session_id == uuid.UUID(session_id)
        ).first()
        if not task:
            return
        task.status = status
        task.progress_pct = pct
        task.progress_message = msg
        if result is not None:
            task.result = result
        if status == "processing" and task.started_at is None:
            task.started_at = datetime.datetime.utcnow()
        if status in ("complete", "failed"):
            task.completed_at = datetime.datetime.utcnow()


def _fail(session_id: str, error_msg: str) -> dict:
    with get_db() as db:
        task = db.query(Task).filter(
            Task.session_id == uuid.UUID(session_id)
        ).first()
        if task:
            task.status = "failed"
            task.progress_pct = 100
            task.progress_message = "Analysis failed"
            task.error = error_msg
            task.completed_at = datetime.datetime.utcnow()
        session = db.query(UserSession).filter(
            UserSession.session_id == uuid.UUID(session_id)
        ).first()
        if session:
            session.status = "failed"

    log.warning("analyze_resume_failed", session_id=session_id, error=error_msg)
    return {"error": error_msg, "session_id": session_id}
