"""
Gap analysis: compute skill gap between user's extracted skills and target role profile.

Readiness score = sum(importance of user's matching skills) / sum(all role skills importance)

Adjusted readiness score = same formula but missing skills with adjacent/transferable
relationships receive partial credit (see GAP_MULTIPLIERS), reflecting a recruiter's
real assessment of ramp-up time.

Returns structured gap data:
  - matched_skills   : skills the user has that are in the role profile
  - missing_skills   : high-importance skills the user is missing (sorted by importance)
  - bonus_skills     : user has these but they are not in the role profile (good-to-know)
  - readiness_score  : 0–100 integer (direct keyword matches only)
  - adjusted_readiness_score : 0–100 integer (includes partial credit for adjacent skills)
  - score_contributors : top 5 positive + top 5 negative contributors with gap_status
  - skill_categories : breakdown by category (for the radar chart on the results page)
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.models.role import RoleSkillProfile
from src.models.skill import Skill, SkillCategory
from src.models.user_skill import UserSkill
from src.services.skill_graph import build_user_skill_index, classify_gap_type, classify_via_graph

log = structlog.get_logger(__name__)

# Base credit per gap_status (unchanged from original).
GAP_MULTIPLIERS: dict[str, float] = {
    "strong_match":       1.00,
    "adjacent_expertise": 0.60,
    "transferable":       0.35,
    "related":            0.15,
    "partial_match":      0.08,
    "missing":            0.00,
}

# ── Recruiter-realism scoring additions ───────────────────────────────────────
#
# Three signals that pure skill-count scoring misses:
#   1. Gap-type credit  — tooling gaps (Docker, Redis) are learnable in days;
#      foundational gaps (Algorithms, System Design) take months.  They warrant
#      different base penalties even when both are "missing".
#
#   2. Domain compression — HTML + CSS + React + Next.js + Tailwind is ONE
#      domain gap (Frontend Engineering), not five independent deficiencies.
#      Subsequent missing skills in the same category receive diminishing penalties.
#
#   3. Engineering maturity bonus — Docker + System Design + Open Source +
#      production APIs signal engineering breadth that survives beyond the
#      keyword-matching layer.  A recruiter sees this immediately; the score
#      should reflect it.

# Additional base credit for missing skills by gap type.
# Stacks on top of GAP_MULTIPLIERS[gap_status].
_GAP_TYPE_CREDIT: dict[str, float] = {
    "foundational": 0.00,   # months to build — full penalty
    "domain":       0.05,   # structured study — slight relief
    "tooling":      0.12,   # days to pick up — substantial relief
}

# Extra credit for the Nth missing skill in the same DB category (0-indexed).
# Position 0 = most important in domain → full penalty.
# Position 1 = second in domain → 0.10 extra credit. Etc.
_DOMAIN_COMPRESSION: list[float] = [0.00, 0.10, 0.15, 0.20, 0.20]

# Hard cap: missing skills never credit more than a weak adjacent skill.
_MISSING_MULTIPLIER_CAP: float = 0.45

# Skills whose presence signals engineering maturity beyond keyword matching.
# Values are bonus points added to adjusted_readiness_score.
_MATURITY_SIGNALS: dict[str, int] = {
    "docker":                    2,
    "docker compose":            1,
    "system design":             4,
    "software architecture":     4,
    "distributed systems":       4,
    "kubernetes":                2,
    "github actions":            1,
    "ci/cd":                     1,
    "open source contribution":  4,
    "rest api":                  2,
    "api development":           2,
    "fastapi":                   2,   # production-grade async Python API — strong backend signal
    "microservices":             3,
    "linux":                     1,
    "postgresql":                1,
    "mysql":                     1,
    "terraform":                 1,
    "compiler design":           3,
    "machine learning":          2,
    "deep learning":             2,
    "git":                       1,
    "python":                    1,   # scripting vs. engineering distinction is made by context
}
_MATURITY_BONUS_CAP: int = 10


def _effective_multiplier(gap_status: str, gap_type: str, domain_position: int) -> float:
    """
    Compute the adjusted-score multiplier for one skill.

    Matched (strong_match) always returns 1.0.
    For non-matched skills the multiplier is:
        GAP_MULTIPLIERS[gap_status]           (base adjacency credit)
      + _GAP_TYPE_CREDIT[gap_type]            (tooling gaps are cheaper to close)
      + _DOMAIN_COMPRESSION[domain_position]  (diminishing penalty for stacked gaps)
    capped at _MISSING_MULTIPLIER_CAP.
    """
    base = GAP_MULTIPLIERS.get(gap_status, 0.0)
    if gap_status == "strong_match":
        return base
    type_credit  = _GAP_TYPE_CREDIT.get(gap_type, _GAP_TYPE_CREDIT["tooling"])
    compression  = _DOMAIN_COMPRESSION[min(domain_position, len(_DOMAIN_COMPRESSION) - 1)]
    return min(base + type_credit + compression, _MISSING_MULTIPLIER_CAP)


def _maturity_bonus(matched_skills: list[dict]) -> int:
    """
    Sum maturity-signal bonus points for the candidate's matched skills.
    Capped at _MATURITY_BONUS_CAP to prevent inflation on near-complete matches.
    """
    matched_names = {s["display_name"].strip().lower() for s in matched_skills}
    total = sum(pts for sig, pts in _MATURITY_SIGNALS.items() if sig in matched_names)
    return min(total, _MATURITY_BONUS_CAP)


def analyze_gap(session_id: str, role_id: int, db: Session) -> dict[str, Any]:
    """
    Compute the skill gap for a given session against a target role.

    Returns:
        {
            "readiness_score": int,
            "adjusted_readiness_score": int,
            "matched_skills": [...],
            "missing_skills": [...],
            "bonus_skills": [...],
            "category_breakdown": {...},
            "score_contributors": [...],
        }
    """
    role_profile = _get_role_profile(role_id, db)
    if not role_profile:
        log.warning("gap_analyze_empty_profile", role_id=role_id)
        return _empty_result()

    user_skill_ids = _get_user_skill_ids(session_id, db)

    total_importance = sum(s["importance_score"] for s in role_profile)
    matched_importance = sum(
        s["importance_score"] for s in role_profile if s["skill_id"] in user_skill_ids
    )
    readiness_score = (
        round((matched_importance / total_importance) * 100) if total_importance > 0 else 0
    )

    matched: list[dict] = []
    missing: list[dict] = []
    role_skill_id_set = {s["skill_id"] for s in role_profile}

    for skill_data in role_profile:
        entry = {
            "skill_id":       skill_data["skill_id"],
            "display_name":   skill_data["display_name"],
            "category":       skill_data["category"],
            "importance_score": skill_data["importance_score"],
            "frequency":      skill_data["frequency"],
            "category_id":    skill_data.get("category_id"),
            "parent_category_id": skill_data.get("parent_category_id"),
        }
        if skill_data["skill_id"] in user_skill_ids:
            entry["confidence"] = user_skill_ids[skill_data["skill_id"]]
            matched.append(entry)
        else:
            missing.append(entry)

    # ── Gap status: skill graph takes priority over DB category hierarchy ────
    #
    # Priority order:
    #   1. Skill graph adjacency  → "adjacent_expertise"  (same paradigm, different tool)
    #   2. Skill graph transfer   → "transferable"         (higher abstraction implies this)
    #   3. Same DB category       → "related"              (same domain, different tool)
    #   4. Sibling DB category    → "partial_match"        (neighbouring domain)
    #   5. No relationship        → "missing"
    #
    # All user skills (matched + bonus) are included in the adjacency index.
    # A skill the user has that is not in this role's profile (e.g. "REST API" for
    # a Java-centric Backend Engineer profile) must still be able to provide
    # adjacent_expertise credit to related missing skills (e.g. Spring Boot).
    all_user_display = _get_skill_display_bulk(list(user_skill_ids.keys()), db)
    user_skill_index = build_user_skill_index(list(all_user_display.values()))
    matched_cat_names: set[str] = {s["category"] for s in matched}

    for s in matched:
        s["gap_status"] = "strong_match"
        s["via_skill"] = None

    for s in missing:
        cat_name = s.get("category", "Other")
        parent_cat_id = s.get("parent_category_id")

        graph_status, via_skill = classify_via_graph(s["display_name"], user_skill_index)
        if graph_status:
            s["gap_status"] = graph_status
            s["via_skill"] = via_skill
        elif cat_name in matched_cat_names:
            s["gap_status"] = "related"
            s["via_skill"] = None
        elif parent_cat_id is not None:
            sibling_cats = {
                rp["category"] for rp in role_profile
                if rp.get("parent_category_id") == parent_cat_id
                and rp["category"] != cat_name
            }
            s["gap_status"] = "partial_match" if sibling_cats & matched_cat_names else "missing"
            s["via_skill"] = None
        else:
            s["gap_status"] = "missing"
            s["via_skill"] = None

        s["gap_type"] = classify_gap_type(s["display_name"])

    # ── Domain compression ────────────────────────────────────────────────────
    # missing is already sorted by importance DESC (from role_profile query).
    # Track position of each skill within its domain; higher positions receive
    # diminishing penalty so stacked gaps in the same domain don't collapse the score.
    _domain_pos: dict[str, int] = {}
    for s in missing:
        domain = s.get("category", "Other")
        pos = _domain_pos.get(domain, 0)
        s["_domain_position"] = pos
        _domain_pos[domain] = pos + 1

    # ── Adjusted readiness: domain compression + gap-type credit + maturity ──
    adjusted_importance = sum(
        s["importance_score"] * _effective_multiplier(
            s.get("gap_status", "strong_match"),
            s.get("gap_type", "tooling"),
            s.get("_domain_position", 0),
        )
        for s in matched + missing
    )
    maturity_pts = _maturity_bonus(matched)
    adjusted_readiness_score = min(100, (
        round((adjusted_importance / total_importance) * 100) if total_importance > 0 else 0
    ) + maturity_pts)

    score_contributors = _build_score_contributors(matched + missing, user_skill_ids, total_importance)

    missing.sort(key=lambda x: x["importance_score"], reverse=True)

    bonus_ids = [sid for sid in user_skill_ids if sid not in role_skill_id_set]
    bonus_display = _get_skill_display_bulk(bonus_ids, db)
    bonus = [
        {
            "skill_id":     sid,
            "display_name": bonus_display.get(sid, f"Skill #{sid}"),
            "confidence":   user_skill_ids[sid],
        }
        for sid in bonus_ids
    ]

    category_breakdown = _build_category_breakdown(matched, missing)

    log.info(
        "gap_analyzed",
        session_id=str(session_id),
        role_id=role_id,
        readiness_score=readiness_score,
        adjusted_readiness_score=adjusted_readiness_score,
        matched=len(matched),
        missing=len(missing),
    )

    return {
        "readiness_score":          readiness_score,
        "adjusted_readiness_score": adjusted_readiness_score,
        "maturity_bonus":           maturity_pts,
        "matched_skills":           matched,
        "missing_skills":           missing,
        "bonus_skills":             bonus,
        "category_breakdown":       category_breakdown,
        "score_contributors":       score_contributors,
        "total_importance":         total_importance,
        "user_skill_ids":           user_skill_ids,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_role_profile(role_id: int, db: Session) -> list[dict]:
    rows = db.execute(
        text(
            """
            SELECT rsp.skill_id, rsp.importance_score, rsp.frequency,
                   s.display_name, sc.name AS category,
                   sc.category_id, sc.parent_id AS parent_category_id
            FROM role_skill_profiles rsp
            JOIN skills s ON s.skill_id = rsp.skill_id
            LEFT JOIN skill_categories sc ON sc.category_id = s.category_id
            WHERE rsp.role_id = :role_id AND s.is_active = true
            ORDER BY rsp.importance_score DESC
            """
        ),
        {"role_id": role_id},
    ).fetchall()

    return [
        {
            "skill_id":          r.skill_id,
            "importance_score":  float(r.importance_score),
            "frequency":         float(r.frequency),
            "display_name":      r.display_name,
            "category":          r.category or "Other",
            "category_id":       r.category_id,
            "parent_category_id": r.parent_category_id,
        }
        for r in rows
    ]


def _get_user_skill_ids(session_id: str, db: Session) -> dict[int, float]:
    rows = db.execute(
        text("SELECT skill_id, confidence FROM user_skills WHERE session_id = :sid"),
        {"sid": str(session_id)},
    ).fetchall()
    return {r.skill_id: float(r.confidence) for r in rows}


def _get_skill_display_bulk(skill_ids: list[int], db: Session) -> dict[int, str]:
    if not skill_ids:
        return {}
    skills = db.query(Skill).filter(Skill.skill_id.in_(skill_ids)).all()
    return {s.skill_id: s.display_name for s in skills}


def _build_category_breakdown(matched: list[dict], missing: list[dict]) -> dict[str, dict]:
    breakdown: dict[str, dict] = {}
    for skill in matched:
        cat = skill["category"]
        breakdown.setdefault(cat, {"matched": 0, "missing": 0})
        breakdown[cat]["matched"] += 1
    for skill in missing:
        cat = skill["category"]
        breakdown.setdefault(cat, {"matched": 0, "missing": 0})
        breakdown[cat]["missing"] += 1
    for cat, data in breakdown.items():
        total = data["matched"] + data["missing"]
        data["total"] = total
        data["pct"] = round(data["matched"] / total * 100) if total else 0
    return breakdown


def _build_score_contributors(
    all_skills: list[dict],
    user_skill_ids: dict[int, float],
    total_importance: float,
) -> list[dict]:
    """
    Return top 5 positive (matched) + top 5 negative (missing) contributors.
    Each entry includes gap_status and via_skill so the UI can show "why."
    """
    if total_importance == 0:
        return []
    contribs = [
        {
            "display_name": s["display_name"],
            "impact":       round((s["importance_score"] / total_importance) * 100),
            "matched":      s["skill_id"] in user_skill_ids,
            "gap_status":   s.get("gap_status", "strong_match" if s["skill_id"] in user_skill_ids else "missing"),
            "via_skill":    s.get("via_skill"),
        }
        for s in all_skills
    ]
    positive = sorted((c for c in contribs if c["matched"]),     key=lambda x: -x["impact"])[:5]
    negative = sorted((c for c in contribs if not c["matched"]), key=lambda x: -x["impact"])[:5]
    return positive + negative


def analyze_gap_jd(session_id: str, jd_skills: list[dict], db: Session) -> dict[str, Any]:
    """
    Compute skill gap between user's resume skills and skills extracted from a JD.

    jd_skills: list of {skill_id, importance_score, display_name, category}
    Uses the same return shape as analyze_gap() so the results page is unchanged.
    """
    if not jd_skills:
        log.warning("gap_analyze_jd_empty_profile", session_id=str(session_id))
        return _empty_result()

    user_skill_ids = _get_user_skill_ids(session_id, db)
    jd_skill_map = {s["skill_id"]: s for s in jd_skills}

    total = len(jd_skills)
    matched_count = sum(1 for sid in jd_skill_map if sid in user_skill_ids)
    readiness_score = round((matched_count / total) * 100) if total > 0 else 0

    matched: list[dict] = []
    missing: list[dict] = []

    for skill_data in jd_skills:
        entry = {
            "skill_id":       skill_data["skill_id"],
            "display_name":   skill_data["display_name"],
            "category":       skill_data.get("category", "Other"),
            "importance_score": skill_data["importance_score"],
            "frequency":      1.0,
        }
        if skill_data["skill_id"] in user_skill_ids:
            entry["confidence"] = user_skill_ids[skill_data["skill_id"]]
            matched.append(entry)
        else:
            missing.append(entry)

    # Gap status for JD mode (no parent_category_id available from JD extraction)
    user_skill_index_jd = build_user_skill_index([s["display_name"] for s in matched])
    matched_cat_names_jd: set[str] = {s.get("category", "Other") for s in matched}

    for s in matched:
        s["gap_status"] = "strong_match"
        s["via_skill"] = None

    for s in missing:
        cat = s.get("category", "Other")
        graph_status, via_skill = classify_via_graph(s["display_name"], user_skill_index_jd)
        if graph_status:
            s["gap_status"] = graph_status
            s["via_skill"] = via_skill
        elif cat in matched_cat_names_jd:
            s["gap_status"] = "related"
            s["via_skill"] = None
        else:
            s["gap_status"] = "missing"
            s["via_skill"] = None

        s["gap_type"] = classify_gap_type(s["display_name"])

    # Domain compression for JD mode (same logic as role-based mode)
    _jd_domain_pos: dict[str, int] = {}
    for s in sorted(missing, key=lambda x: x["importance_score"], reverse=True):
        domain = s.get("category", "Other")
        pos = _jd_domain_pos.get(domain, 0)
        s["_domain_position"] = pos
        _jd_domain_pos[domain] = pos + 1

    # Adjusted readiness for JD mode with domain compression + gap-type credit
    adjusted_matched = sum(
        _effective_multiplier(
            s.get("gap_status", "strong_match"),
            s.get("gap_type", "tooling"),
            s.get("_domain_position", 0),
        )
        for s in matched + missing
    )
    jd_maturity_pts = _maturity_bonus(matched)
    adjusted_readiness_score = min(100, (
        round((adjusted_matched / total) * 100) if total > 0 else 0
    ) + jd_maturity_pts)

    per_impact = round(100 / total) if total > 0 else 0
    score_contributors = (
        [{"display_name": s["display_name"], "impact": per_impact, "matched": True,
          "gap_status": "strong_match", "via_skill": None}
         for s in matched][:5]
        + [{"display_name": s["display_name"], "impact": per_impact, "matched": False,
            "gap_status": s.get("gap_status", "missing"), "via_skill": s.get("via_skill")}
           for s in missing][:5]
    )

    missing.sort(key=lambda x: x["importance_score"], reverse=True)

    bonus_ids = [sid for sid in user_skill_ids if sid not in jd_skill_map]
    bonus_display = _get_skill_display_bulk(bonus_ids, db)
    bonus = [
        {
            "skill_id":     sid,
            "display_name": bonus_display.get(sid, f"Skill #{sid}"),
            "confidence":   user_skill_ids[sid],
        }
        for sid in bonus_ids
    ]

    category_breakdown = _build_category_breakdown(matched, missing)

    log.info(
        "gap_analyzed_jd",
        session_id=str(session_id),
        readiness_score=readiness_score,
        adjusted_readiness_score=adjusted_readiness_score,
        jd_skills=total,
        matched=len(matched),
        missing=len(missing),
    )

    return {
        "readiness_score":          readiness_score,
        "adjusted_readiness_score": adjusted_readiness_score,
        "maturity_bonus":           jd_maturity_pts,
        "matched_skills":           matched,
        "missing_skills":           missing,
        "bonus_skills":             bonus,
        "category_breakdown":       category_breakdown,
        "score_contributors":       score_contributors,
        "total_importance":         float(total),
        "user_skill_ids":           user_skill_ids,
    }


def build_jd_skill_profile(extracted: list[dict], db: Session) -> list[dict]:
    """
    Convert raw extract_skills() output into a JD skill profile.
    Fetches display_name + category for every extracted skill in one query.
    importance_score = extraction confidence (alias match = 0.95, embedding varies).
    """
    if not extracted:
        return []

    skill_ids = [s["skill_id"] for s in extracted]
    rows = db.execute(
        __import__("sqlalchemy").text(
            """
            SELECT s.skill_id, s.display_name, sc.name AS category
            FROM skills s
            LEFT JOIN skill_categories sc ON sc.category_id = s.category_id
            WHERE s.skill_id = ANY(:ids)
            """
        ),
        {"ids": skill_ids},
    ).fetchall()

    meta = {r.skill_id: {"display_name": r.display_name, "category": r.category or "Other"}
            for r in rows}

    return [
        {
            "skill_id":       s["skill_id"],
            "importance_score": round(s["confidence"], 4),
            "display_name":   meta.get(s["skill_id"], {}).get("display_name", f"Skill #{s['skill_id']}"),
            "category":       meta.get(s["skill_id"], {}).get("category", "Other"),
        }
        for s in extracted
        if s["skill_id"] in meta
    ]


def get_all_role_profiles_bulk(db: Session) -> list[dict]:
    """
    Fetch every role's skill profile in a single query.
    Used by recruiter.py to compute cross-role fit without per-role DB round-trips.
    Returns list of {role_id, display_name, domain, skill_id, skill_name, importance_score}.
    skill_name is included so recruiter.py can apply graph adjacency in the fit ranking.
    """
    rows = db.execute(
        text(
            """
            SELECT rsp.role_id, rsp.skill_id, rsp.importance_score,
                   r.display_name, r.domain,
                   s.display_name AS skill_name
            FROM role_skill_profiles rsp
            JOIN role_categories r ON r.role_id = rsp.role_id
            JOIN skills s ON s.skill_id = rsp.skill_id
            ORDER BY rsp.role_id, rsp.importance_score DESC
            """
        )
    ).fetchall()
    return [
        {
            "role_id":          r.role_id,
            "skill_id":         r.skill_id,
            "importance_score": float(r.importance_score),
            "display_name":     r.display_name,
            "domain":           r.domain or "Other",
            "skill_name":       r.skill_name,
        }
        for r in rows
    ]


def _empty_result() -> dict:
    return {
        "readiness_score":          0,
        "adjusted_readiness_score": 0,
        "maturity_bonus":           0,
        "matched_skills":           [],
        "missing_skills":           [],
        "bonus_skills":             [],
        "category_breakdown":       {},
        "score_contributors":       [],
        "total_importance":         0.0,
        "user_skill_ids":           {},
    }
