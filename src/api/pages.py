"""
Page routes: HTML responses rendered via Jinja2.

Routes:
  GET /                   → landing page (upload form)
  GET /processing/{sid}   → polling page (auto-refreshes until complete)
  GET /results/{sid}      → results page (readiness score + roadmap)
"""

import os
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from src.database import get_db_session as get_db
from src.models.jd import JDAnalysis
from src.models.resume import Resume
from src.models.roadmap import Roadmap
from src.models.session import UserSession
from src.models.task import Task
from src.services.candidate_profiler import build_candidate_profile
from src.services.gap_analyzer import get_all_role_profiles_bulk
from src.services.recruiter import (
    compute_evidence_map,
    compute_project_recommendations,
    compute_recruiter_summary,
    compute_role_fit_ranking,
    compute_shortest_path,
)
from src.services.soft_skill_inferencer import build_skill_evidence, infer_soft_skills

log = structlog.get_logger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="src/templates")
FRONTEND_BASE_URL = os.getenv("NEXT_PUBLIC_FRONTEND_URL") or os.getenv("FRONTEND_BASE_URL") or "http://127.0.0.1:3000"


def _frontend_url(path: str) -> str:
    return f"{FRONTEND_BASE_URL.rstrip('/')}/{path.lstrip('/')}"


def _build_results_context(session_id: str, db: Session) -> dict:
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

    session = db.query(UserSession).filter(UserSession.session_id == sid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status not in ("complete",):
        raise HTTPException(status_code=409, detail="Results are not ready yet")

    if session.is_expired():
        raise HTTPException(status_code=410, detail="This session has expired. Please upload your resume again.")

    roadmap = db.query(Roadmap).filter(Roadmap.session_id == sid).first()
    jd_analysis = db.query(JDAnalysis).filter(JDAnalysis.session_id == sid).first()
    resume = db.query(Resume).filter(Resume.session_id == sid).first()

    from src.services.gap_analyzer import analyze_gap, analyze_gap_jd
    if jd_analysis and jd_analysis.jd_skills:
        gap_data = analyze_gap_jd(str(session_id), jd_analysis.jd_skills, db)
    elif session.role_id:
        gap_data = analyze_gap(str(session_id), session.role_id, db)
    else:
        gap_data = {
            "readiness_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "bonus_skills": [],
            "category_breakdown": {},
            "score_contributors": [],
        }

    sections = resume.sections if resume else {}
    if isinstance(sections, dict) and sections.get("_wiped"):
        soft_skill_inferences = sections.get("_inferences", [])
        raw_evidence = sections.get("_evidence", {})
        skill_evidence = {int(k): v for k, v in raw_evidence.items()}
    else:
        soft_skill_inferences = infer_soft_skills(sections or {})
        skill_evidence = build_skill_evidence(sections or {}, gap_data.get("matched_skills", []))

    candidate_profile = build_candidate_profile(
        gap_data.get("matched_skills", []),
        gap_data.get("missing_skills", []),
        soft_skill_inferences,
    )

    raw_score = session.readiness_score or 0
    adjusted_score = gap_data.get("adjusted_readiness_score", raw_score)

    recruiter_summary = compute_recruiter_summary(gap_data, adjusted_score, candidate_profile) if gap_data else {}
    evidence_map = compute_evidence_map(gap_data) if gap_data else {}
    project_recommendations = compute_project_recommendations(
        gap_data.get("missing_skills", []),
        gap_data.get("matched_skills", []),
        candidate_profile,
    ) if gap_data else []

    all_role_profiles = get_all_role_profiles_bulk(db)
    user_skill_ids_set = set(gap_data.get("user_skill_ids", {}).keys())
    user_skill_names = (
        {s["display_name"] for s in gap_data.get("matched_skills", [])}
        | {s["display_name"] for s in gap_data.get("bonus_skills", [])}
    )
    role_fit = compute_role_fit_ranking(
        user_skill_ids_set,
        all_role_profiles,
        current_role_id=session.role_id,
        user_skill_names=user_skill_names,
    )

    shortest_path = compute_shortest_path(
        gap_data.get("missing_skills", []),
        gap_data.get("total_importance", 0.0),
        adjusted_score,
    )

    return {
        "session": session,
        "session_id": session_id,
        "gap_data": gap_data,
        "roadmap": roadmap.content if roadmap else {},
        "readiness_score": adjusted_score,
        "raw_readiness_score": raw_score,
        "target_role": session.target_role,
        "jd_analysis": jd_analysis,
        "recruiter_summary": recruiter_summary,
        "evidence_map": evidence_map,
        "project_recommendations": project_recommendations,
        "soft_skill_inferences": soft_skill_inferences,
        "skill_evidence": skill_evidence,
        "score_contributors": gap_data.get("score_contributors", []),
        "candidate_profile": candidate_profile,
        "role_fit": role_fit,
        "shortest_path": shortest_path,
    }


def _serialize_results_payload(context: dict) -> dict:
    jd_analysis = context.get("jd_analysis")
    session = context.get("session")

    payload = {
        "session_id": context.get("session_id"),
        "target_role": context.get("target_role"),
        "readiness_score": context.get("readiness_score"),
        "raw_readiness_score": context.get("raw_readiness_score"),
        "gap_data": context.get("gap_data", {}),
        "roadmap": context.get("roadmap", {}),
        "recruiter_summary": context.get("recruiter_summary", {}),
        "evidence_map": context.get("evidence_map", {}),
        "project_recommendations": context.get("project_recommendations", []),
        "soft_skill_inferences": context.get("soft_skill_inferences", []),
        "skill_evidence": context.get("skill_evidence", {}),
        "score_contributors": context.get("score_contributors", []),
        "candidate_profile": context.get("candidate_profile", {}),
        "role_fit": context.get("role_fit", []),
        "shortest_path": context.get("shortest_path", {}),
        "session": {
            "session_id": str(session.session_id) if session else None,
            "target_role": session.target_role if session else None,
            "status": session.status if session else None,
            "readiness_score": session.readiness_score if session else None,
            "expires_at": session.expires_at.isoformat() if getattr(session, "expires_at", None) else None,
        },
        "jd_analysis": {
            "jd_skills": jd_analysis.jd_skills if jd_analysis else None,
            "summary": jd_analysis.summary if jd_analysis else None,
            "confidence": jd_analysis.confidence if jd_analysis else None,
        } if jd_analysis else None,
    }
    return payload


@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return RedirectResponse(url=_frontend_url("/"))


@router.get("/processing/{session_id}", response_class=HTMLResponse)
async def processing_page(
    request: Request,
    session_id: str,
    db: Session = Depends(get_db),
):
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

    session = db.query(UserSession).filter(UserSession.session_id == sid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    task = db.query(Task).filter(Task.session_id == sid).first()

    return RedirectResponse(url=_frontend_url(f"/processing/{session_id}"))


@router.get("/results/{session_id}", response_class=HTMLResponse)
async def results_page(
    request: Request,
    session_id: str,
    db: Session = Depends(get_db),
):
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")

    session = db.query(UserSession).filter(UserSession.session_id == sid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status not in ("complete",):
        return RedirectResponse(url=_frontend_url(f"/processing/{session_id}"))

    if session.is_expired():
        return RedirectResponse(url=_frontend_url(f"/results/{session_id}"))

    return RedirectResponse(url=_frontend_url(f"/results/{session_id}"))


@router.get("/v1/results/{session_id}")
async def results_api(session_id: str, db: Session = Depends(get_db)):
    try:
        _build_results_context(session_id, db)
    except HTTPException as exc:
        if exc.status_code == 409:
            return JSONResponse(status_code=409, content={"detail": exc.detail})
        if exc.status_code == 410:
            return JSONResponse(status_code=410, content={"detail": exc.detail})
        raise

    context = _build_results_context(session_id, db)
    payload = _serialize_results_payload(context)
    return JSONResponse(status_code=200, content=jsonable_encoder(payload))
