"""
Celery task: JD match analysis pipeline.

Steps:
  1. Update task status → "processing" (10%)
  2. Parse resume bytes → raw_text + sections            [Phase 1 DB context]
  3. Extract skills from resume (two-pass)               [Phase 1 DB context]
  4. Extract skills from JD text (reuses same extractor) [Phase 1 DB context]
  5. Build JD skill profile from extracted JD skills     [Phase 1 DB context]
  6. Persist user_skills + jd_skills                     [Phase 1 DB context]
  7. Compute gap (resume skills vs JD skills)            [Phase 1 DB context]
     → Phase 1 DB connection released before LLM call
  8. Update task → 75%, generate roadmap (LLM)           [no DB held]
  9. Update task → 95%                                   [Phase 2 DB context]
 10. Persist roadmap + finalize session                  [Phase 2 DB context]
 11. Update task → complete (100%)
"""

import datetime
import uuid
from typing import Any

import structlog

from src.database import get_db
from src.models.jd import JDAnalysis
from src.models.resume import Resume
from src.models.roadmap import Roadmap
from src.models.session import UserSession
from src.models.task import Task
from src.models.user_skill import UserSkill
from src.services.gap_analyzer import analyze_gap_jd, build_jd_skill_profile
from src.services.resume_parser import ParseError, parse_resume
from src.services.roadmap_generator import generate_roadmap
from src.services.skill_extractor import extract_skills
from src.worker import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.analyze_jd",
    max_retries=2,
    default_retry_delay=10,
    time_limit=300,
    soft_time_limit=270,
)
def analyze_jd_task(
    self,
    session_id: str,
    file_bytes: bytes,
    filename: str,
    jd_text: str,
    job_title: str,
    company_name: str,
    duration: str = "3 months",
) -> dict[str, Any]:
    """
    Full JD matching pipeline. Uses the same services as analyze_resume_task
    but replaces the role_skill_profiles gap analysis with a JD-derived profile.
    """
    _update_task(session_id, status="processing", pct=10, msg="Parsing resume…")

    try:
        log.info("analyze_jd_start", session_id=session_id, filename=filename)

        # ── Phase 1: parse + extract (fast, no LLM) ──────────────────────────
        with get_db() as db:
            try:
                parsed = parse_resume(file_bytes, filename)
            except ParseError as exc:
                return _fail(session_id, str(exc))

            # Persist resume stub
            resume = db.query(Resume).filter(
                Resume.session_id == uuid.UUID(session_id)
            ).first()
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

            _update_task(session_id, status="processing", pct=35, msg="Extracting resume skills…")

            # Extract skills from resume
            resume_skills = extract_skills(parsed["sections"], parsed["raw_text"], db)

            # Persist user_skills
            db.query(UserSkill).filter(
                UserSkill.session_id == uuid.UUID(session_id)
            ).delete()
            for s in resume_skills:
                db.add(UserSkill(
                    session_id=uuid.UUID(session_id),
                    skill_id=s["skill_id"],
                    confidence=s["confidence"],
                    source=s["source"],
                ))
            db.flush()

            _update_task(session_id, status="processing", pct=55, msg="Matching against job description…")

            # Extract skills from JD — treat the whole JD as a "skills" section
            # so every mention gets the 3× confidence weight (all JD skills are explicit requirements).
            jd_sections = {"skills": jd_text}
            jd_extracted = extract_skills(jd_sections, jd_text, db)

            # Build typed JD skill profile (adds display_name + category)
            jd_profile = build_jd_skill_profile(jd_extracted, db)

            # ── F5: Expand JD concepts into implied skill names ────────────────
            # Phrases like "scalable ML systems" imply MLOps, Docker, etc.
            # Added at importance 0.4 — implied requirements, not explicit keywords.
            from sqlalchemy import text as _sql
            from src.services.jd_expander import expand_jd_concepts
            expanded_names = expand_jd_concepts(jd_text)
            if expanded_names:
                existing_ids = {s["skill_id"] for s in jd_profile}
                for skill_name in expanded_names:
                    row = db.execute(
                        _sql(
                            "SELECT skill_id FROM skills "
                            "WHERE LOWER(display_name) = LOWER(:name) AND is_active = true "
                            "LIMIT 1"
                        ),
                        {"name": skill_name},
                    ).first()
                    if row and row.skill_id not in existing_ids:
                        jd_profile.append({
                            "skill_id": row.skill_id,
                            "importance_score": 0.4,
                            "display_name": skill_name,
                            "category": "Other",
                        })
                        existing_ids.add(row.skill_id)

            # Persist JD skills into jd_analyses record
            jd_record = db.query(JDAnalysis).filter(
                JDAnalysis.session_id == uuid.UUID(session_id)
            ).first()
            if jd_record:
                jd_record.jd_skills = jd_profile
            db.flush()

            # Compute gap: resume skills vs JD-derived profile
            gap_data = analyze_gap_jd(session_id, jd_profile, db)

            session = db.query(UserSession).filter(
                UserSession.session_id == uuid.UUID(session_id)
            ).first()
            if session:
                session.status = "processing"
            db.flush()

            # Phase 1 commits and releases connection before LLM call

        # ── 75%: roadmap generation (outside DB context) ──────────────────────
        _update_task(session_id, status="processing", pct=75, msg="Generating targeted roadmap…")

        display_name = job_title or "the target role"
        if company_name:
            display_name = f"{job_title} at {company_name}" if job_title else f"role at {company_name}"

        roadmap_result = generate_roadmap(
            session_id=session_id,
            role_display_name=display_name,
            gap_data=gap_data,
            duration=duration,
        )

        # ── Phase 2: persist ──────────────────────────────────────────────────
        with get_db() as db:
            existing = db.query(Roadmap).filter(
                Roadmap.session_id == uuid.UUID(session_id)
            ).first()
            if existing:
                existing.content = roadmap_result["content"]
                existing.model_used = roadmap_result["model_used"]
                existing.generation_ms = roadmap_result["generation_ms"]
                existing.cache_hit = roadmap_result["cache_hit"]
                existing.prompt_version = roadmap_result["prompt_version"]
            else:
                db.add(Roadmap(
                    session_id=uuid.UUID(session_id),
                    role_id=None,   # JD mode — no canonical role
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

            # ── F9 + F3 + F1: cache evidence/inferences before TTL wipe ──────
            from src.services.soft_skill_inferencer import (
                build_skill_evidence,
                infer_soft_skills,
            )
            resume_rec = db.query(Resume).filter(
                Resume.session_id == uuid.UUID(session_id)
            ).first()
            if resume_rec and isinstance(resume_rec.sections, dict):
                inferences = infer_soft_skills(resume_rec.sections)
                evidence = build_skill_evidence(
                    resume_rec.sections, gap_data.get("matched_skills", [])
                )
                resume_rec.sections = {
                    **resume_rec.sections,
                    "_inferences": inferences,
                    "_evidence": {str(k): v for k, v in evidence.items()},
                }

        # ── F9: Schedule privacy wipe — 15 minutes TTL ───────────────────────
        from src.tasks.privacy import wipe_resume_text
        wipe_resume_text.apply_async(args=[session_id], countdown=900)

        result_payload = {
            "session_id": session_id,
            "readiness_score": gap_data["readiness_score"],
            "matched_skills_count": len(gap_data["matched_skills"]),
            "missing_skills_count": len(gap_data["missing_skills"]),
        }

        _update_task(
            session_id, status="complete", pct=100,
            msg="Analysis complete!", result=result_payload,
        )

        log.info(
            "analyze_jd_complete",
            session_id=session_id,
            readiness_score=gap_data["readiness_score"],
            jd_skills=len(jd_profile),
        )
        return result_payload

    except Exception as exc:
        log.exception("analyze_jd_unhandled_error", session_id=session_id)
        return _fail(session_id, f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Helpers (same pattern as analyze_resume.py)
# ---------------------------------------------------------------------------

def _update_task(
    session_id: str,
    status: str,
    pct: int,
    msg: str,
    result: dict | None = None,
) -> None:
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

    log.warning("analyze_jd_failed", session_id=session_id, error=error_msg)
    return {"error": error_msg, "session_id": session_id}
