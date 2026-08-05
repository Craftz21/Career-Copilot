"""
Phase 4 — Scoring Sanity Tests (Expected Ordering)

For each candidate persona, the role ranking must produce a sane ordering.
These tests catch regressions where a scoring change breaks role ordering.

Expected orderings (from the spec):
  My resume (ML/Backend hybrid):
    ML Engineer >= AI Engineer >= Backend Engineer > Frontend Engineer

  Backend resume:
    Backend Engineer > AI Engineer (backend skills score higher for backend role)

  Frontend resume:
    Frontend Engineer > Backend Engineer

  Data Scientist resume:
    Data Scientist > Backend Engineer

Implementation:
  Uses compute_role_fit_ranking() — the same engine used in the results page.
  Seeds a test UserSession + user_skills in the transactional session (auto-rollback).
  Requires seeded role_categories + skills + role_skill_profiles.
"""

import pytest

from src.services.gap_analyzer import get_all_role_profiles_bulk
from src.services.recruiter import compute_role_fit_ranking

from .helpers import (
    RESUME_BACKEND,
    RESUME_DATA_SCIENTIST,
    RESUME_FRONTEND,
    RESUME_ML,
    RESUME_MY,
    create_test_session,
    get_role_id,
    get_skill_ids,
    make_pdf_bytes,
    seed_user_skills,
)


# ---------------------------------------------------------------------------
# Skill sets per persona (display names matching the DB)
# ---------------------------------------------------------------------------

_SKILLS_MY = [
    "Python", "FastAPI", "REST API", "PostgreSQL", "Docker", "Redis",
    "Git", "Linux", "SQL", "PyTorch", "TensorFlow", "Machine Learning",
    "Deep Learning", "ETL Pipelines", "Async Programming",
]

_SKILLS_BACKEND = [
    "Python", "FastAPI", "PostgreSQL", "Docker", "Redis", "REST API",
    "SQL", "Linux", "Git", "GitHub Actions", "Celery", "SQLAlchemy",
]

_SKILLS_FRONTEND = [
    "JavaScript", "TypeScript", "React", "Next.js", "Vue.js",
    "CSS", "HTML", "Tailwind CSS", "Jest", "Webpack", "Git",
]

_SKILLS_DATA_SCIENTIST = [
    "Python", "R", "SQL", "Pandas", "NumPy", "scikit-learn",
    "Tableau", "Machine Learning", "Statistical Analysis",
]


# ---------------------------------------------------------------------------
# Fixture: all role profiles loaded once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_role_profiles(eval_session):
    return get_all_role_profiles_bulk(eval_session)


# ---------------------------------------------------------------------------
# Helper: build fit ranking for a skill list
# ---------------------------------------------------------------------------

def _rank(skill_names: list[str], db, all_profiles: list[dict]) -> dict[str, int]:
    """
    Return {role_display_name: fit_pct} for a candidate with the given skills.
    Looks up skill_ids from DB; silently ignores unknown skill names.
    """
    from src.services.skill_extractor import _alias_scan

    # Get canonical IDs for the skill names (partial match is fine)
    skill_id_map = get_skill_ids(skill_names, db)
    user_skill_ids = set(skill_id_map.values())
    user_skill_names = set(skill_id_map.keys())

    if not user_skill_ids:
        return {}

    result = compute_role_fit_ranking(
        user_skill_ids,
        all_profiles,
        user_skill_names=user_skill_names,
    )
    all_results = (
        result.get("apply_now", [])
        + result.get("need_experience", [])
        + result.get("farther_away", [])
    )
    return {r["display_name"]: r["fit_pct"] for r in all_results}


def _role_score(rankings: dict[str, int], *names: str) -> int:
    """Return best score for a role across multiple candidate display_names."""
    for name in names:
        if name in rankings:
            return rankings[name]
    return 0


# ---------------------------------------------------------------------------
# Phase 4-A: My resume ordering
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestMyResumeOrdering:
    def test_ml_engineer_gte_backend_engineer(self, eval_session, all_role_profiles):
        rankings = _rank(_SKILLS_MY, eval_session, all_role_profiles)
        ml_score  = _role_score(rankings, "ML Engineer", "Machine Learning Engineer")
        be_score  = _role_score(rankings, "Backend Software Engineer", "Backend Engineer")

        if ml_score == 0 or be_score == 0:
            pytest.skip("Required roles not found in seeded DB — check role display names")

        assert ml_score >= be_score, (
            f"My resume: ML Engineer ({ml_score}%) < Backend Engineer ({be_score}%). "
            "ML/AI skills should score higher for ML role."
        )

    def test_backend_engineer_gt_frontend_engineer(self, eval_session, all_role_profiles):
        rankings  = _rank(_SKILLS_MY, eval_session, all_role_profiles)
        be_score  = _role_score(rankings, "Backend Software Engineer", "Backend Engineer")
        fe_score  = _role_score(rankings, "Frontend Developer", "Frontend Engineer")

        if be_score == 0 or fe_score == 0:
            pytest.skip("Required roles not found in seeded DB")

        assert be_score > fe_score, (
            f"My resume: Backend ({be_score}%) not > Frontend ({fe_score}%). "
            "Resume has no React/CSS/JS — frontend should score lower."
        )

    def test_ai_engineer_gte_backend_engineer(self, eval_session, all_role_profiles):
        rankings  = _rank(_SKILLS_MY, eval_session, all_role_profiles)
        ai_score  = _role_score(rankings, "AI Engineer")
        be_score  = _role_score(rankings, "Backend Software Engineer", "Backend Engineer")

        if ai_score == 0 or be_score == 0:
            pytest.skip("Required roles not found in seeded DB")

        assert ai_score >= be_score, (
            f"My resume: AI Engineer ({ai_score}%) < Backend Engineer ({be_score}%). "
            "PyTorch + TensorFlow + ML should push AI above pure backend."
        )


# ---------------------------------------------------------------------------
# Phase 4-B: Backend resume ordering
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestBackendResumeOrdering:
    def test_backend_engineer_gt_ai_engineer(self, eval_session, all_role_profiles):
        """
        A pure backend resume (no ML skills) must score higher for Backend Engineer
        than for AI Engineer. Fails if adjacency over-credits unrelated domains.
        """
        rankings = _rank(_SKILLS_BACKEND, eval_session, all_role_profiles)
        be_score = _role_score(rankings, "Backend Software Engineer", "Backend Engineer")
        ai_score = _role_score(rankings, "AI Engineer")

        if be_score == 0 or ai_score == 0:
            pytest.skip("Required roles not found in seeded DB")

        assert be_score > ai_score, (
            f"Backend resume: Backend ({be_score}%) not > AI ({ai_score}%). "
            "Backend skills should dominate for backend role; no ML skills present."
        )

    def test_backend_engineer_gt_frontend_engineer(self, eval_session, all_role_profiles):
        rankings = _rank(_SKILLS_BACKEND, eval_session, all_role_profiles)
        be_score = _role_score(rankings, "Backend Software Engineer", "Backend Engineer")
        fe_score = _role_score(rankings, "Frontend Developer", "Frontend Engineer")

        if be_score == 0 or fe_score == 0:
            pytest.skip("Required roles not found in seeded DB")

        assert be_score > fe_score, (
            f"Backend resume: Backend ({be_score}%) not > Frontend ({fe_score}%). "
            "No React/CSS/JS — frontend should score lower."
        )


# ---------------------------------------------------------------------------
# Phase 4-C: Frontend resume ordering
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestFrontendResumeOrdering:
    def test_frontend_gt_backend(self, eval_session, all_role_profiles):
        rankings = _rank(_SKILLS_FRONTEND, eval_session, all_role_profiles)
        fe_score = _role_score(rankings, "Frontend Developer", "Frontend Engineer")
        be_score = _role_score(rankings, "Backend Software Engineer", "Backend Engineer")

        if fe_score == 0 or be_score == 0:
            pytest.skip("Required roles not found in seeded DB")

        assert fe_score > be_score, (
            f"Frontend resume: Frontend ({fe_score}%) not > Backend ({be_score}%). "
            "React/JS/CSS should dominate for a frontend role."
        )


# ---------------------------------------------------------------------------
# Phase 4-D: Data Scientist resume ordering
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestDataScienceOrdering:
    def test_data_scientist_gt_backend(self, eval_session, all_role_profiles):
        rankings = _rank(_SKILLS_DATA_SCIENTIST, eval_session, all_role_profiles)
        ds_score = _role_score(rankings, "Data Scientist")
        be_score = _role_score(rankings, "Backend Software Engineer", "Backend Engineer")

        if ds_score == 0 or be_score == 0:
            pytest.skip("Required roles not found in seeded DB")

        assert ds_score > be_score, (
            f"Data Scientist resume: DS ({ds_score}%) not > Backend ({be_score}%). "
            "Stats + ML skills should dominate for data science role."
        )


# ---------------------------------------------------------------------------
# Phase 4-E: Score magnitude sanity
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestScoreMagnitude:
    def test_perfect_match_approaches_100(self, eval_session, all_role_profiles):
        """
        Seeding every role skill into user_skills should produce near-100% fit.
        Tests that the score ceiling is actually reachable.
        """
        if not all_role_profiles:
            pytest.skip("No role profiles — seed DB first")

        # Grab the first role and all its skills
        first_role_id   = all_role_profiles[0]["role_id"]
        first_role_name = all_role_profiles[0]["display_name"]
        role_skill_ids  = {r["skill_id"] for r in all_role_profiles if r["role_id"] == first_role_id}
        skill_names     = {r["skill_name"] for r in all_role_profiles if r["role_id"] == first_role_id}

        result = compute_role_fit_ranking(
            role_skill_ids,
            all_role_profiles,
            user_skill_names=skill_names,
        )
        all_results = (
            result.get("apply_now", [])
            + result.get("need_experience", [])
            + result.get("farther_away", [])
        )
        scores = {r["display_name"]: r["fit_pct"] for r in all_results}
        fit_for_target = scores.get(first_role_name, 0)

        # With all skills present, score must be >= 80% (maturity bonus may push it higher)
        assert fit_for_target >= 80, (
            f"Seeding all skills for {first_role_name!r} produced only {fit_for_target}%. "
            "Scoring engine may have a bug that prevents high scores."
        )

    def test_no_skills_produces_low_score(self, eval_session, all_role_profiles):
        """A candidate with zero skills should score near 0% for every role."""
        result = compute_role_fit_ranking(
            set(),  # no skill IDs
            all_role_profiles,
            user_skill_names=set(),
        )
        all_results = (
            result.get("apply_now", [])
            + result.get("need_experience", [])
            + result.get("farther_away", [])
        )
        for r in all_results:
            assert r["fit_pct"] <= 10, (
                f"Zero-skill candidate scored {r['fit_pct']}% for {r['display_name']!r}. "
                "Scoring has a non-zero floor that inflates empty resumes."
            )
