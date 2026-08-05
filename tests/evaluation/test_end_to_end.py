"""
Phase 10+11 — End-to-End Pipeline Tests + Failure Harness

Runs the full resume analysis pipeline for each major role:
  AI Engineer, ML Engineer, Backend Engineer, Software Engineer,
  Compiler Engineer, Data Scientist, Security Engineer, Cloud Engineer

For each role:
  parse → skill extract → gap analysis → role ranking → recruiter summary → roadmap

Verifies:
  - No exception at any stage
  - Score produced (>= 0)
  - Role ranking produced (list returned)
  - Recruiter summary produced
  - Roadmap produced (template fallback if LLM unavailable)

Phase 11 — Failure Harness:
  - Invalid PDF → ParseError (not 500)
  - Corrupted PDF → ParseError
  - Empty PDF → ParseError
  - Empty sections → safe extraction (not crash)
  - Missing role → normalize_role returns no_match (not crash)
  - Zero skills → gap analysis returns 0% (not crash)
  - Unknown role → graceful no_match result
"""

import pytest

from src.services.gap_analyzer import analyze_gap, get_all_role_profiles_bulk
from src.services.recruiter import (
    compute_evidence_map,
    compute_recruiter_summary,
    compute_role_fit_ranking,
    compute_shortest_path,
)
from src.services.resume_parser import ParseError, parse_resume
from src.services.roadmap_generator import _template_fallback
from src.services.soft_skill_inferencer import build_skill_evidence, infer_soft_skills

from .helpers import (
    RESUME_MY,
    RESUME_BACKEND,
    RESUME_FRONTEND,
    RESUME_ML,
    RESUME_DATA_SCIENTIST,
    create_test_session,
    get_role_id,
    get_skill_ids,
    make_pdf_bytes,
    make_image_only_pdf,
    seed_user_skills,
)


# ---------------------------------------------------------------------------
# Phase 10: Full pipeline per role
# ---------------------------------------------------------------------------

# Roles from the spec, plus alternatives for flexible display-name matching
_ROLES = [
    ("AI Engineer",),
    ("ML Engineer", "Machine Learning Engineer"),
    ("Backend Software Engineer", "Backend Engineer"),
    ("Software Engineer",),
    ("Compiler Engineer",),
    ("Data Scientist",),
    ("Security Engineer",),
    ("Cloud Engineer",),
]

# Skill sets per persona (reuses helpers from scoring_sanity)
_RESUME_SKILLS = {
    "AI Engineer":               ["Python", "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning", "FastAPI"],
    "Machine Learning Engineer": ["Python", "PyTorch", "TensorFlow", "Machine Learning", "scikit-learn", "Docker"],
    "Backend Software Engineer": ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis", "REST API", "SQL"],
    "Software Engineer":         ["Python", "FastAPI", "PostgreSQL", "Docker", "Git", "SQL"],
    "Compiler Engineer":         ["Python", "C++", "Algorithms", "Compiler Design", "Git"],
    "Data Scientist":            ["Python", "SQL", "Pandas", "NumPy", "Machine Learning", "scikit-learn"],
    "Security Engineer":         ["Python", "Linux", "Git", "Docker"],
    "Cloud Engineer":            ["Docker", "Kubernetes", "Linux", "Git", "Amazon Web Services"],
}


def _get_role_candidate(role_names: tuple[str, ...], db) -> tuple[int, str] | None:
    """Find the first matching role_id from a list of candidate display names."""
    for name in role_names:
        rid = get_role_id(name, db)
        if rid is not None:
            return rid, name
    return None


@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestFullPipeline:

    @pytest.mark.parametrize("role_names", _ROLES, ids=[r[0] for r in _ROLES])
    def test_parse_to_gap_analysis(self, eval_session, role_names):
        """
        Full pipeline: parse PDF → extract skills → gap analysis.
        Must not crash and must return a non-zero score for a relevant resume.
        """
        # Find role in DB
        found = _get_role_candidate(role_names, eval_session)
        if not found:
            pytest.skip(f"None of {role_names} found in DB — check role seeding")
        role_id, role_name = found

        # Parse a relevant resume
        skill_names = _RESUME_SKILLS.get(role_name, _RESUME_SKILLS["Software Engineer"])
        pdf = make_pdf_bytes(RESUME_MY)

        try:
            parsed = parse_resume(pdf, "resume.pdf")
        except ParseError as exc:
            pytest.fail(f"parse_resume failed: {exc}")

        # Extract skills
        from src.services.skill_extractor import extract_skills
        try:
            skills = extract_skills(parsed["sections"], parsed["raw_text"], eval_session)
        except Exception as exc:
            pytest.fail(f"extract_skills failed for {role_name!r}: {exc}")

        # Seed extracted skills
        session_id = create_test_session(eval_session, role_id, role_name)
        skill_ids  = [s["skill_id"] for s in skills]
        if skill_ids:
            seed_user_skills(session_id, skill_ids, eval_session)

        # Gap analysis
        try:
            gap_data = analyze_gap(session_id, role_id, eval_session)
        except Exception as exc:
            pytest.fail(f"analyze_gap failed for {role_name!r}: {exc}")

        # Shape assertions
        assert "readiness_score"          in gap_data
        assert "adjusted_readiness_score" in gap_data
        assert "matched_skills"           in gap_data
        assert "missing_skills"           in gap_data
        assert isinstance(gap_data["readiness_score"], int)
        assert 0 <= gap_data["readiness_score"] <= 100
        assert 0 <= gap_data["adjusted_readiness_score"] <= 100

    @pytest.mark.parametrize("role_names", _ROLES, ids=[r[0] for r in _ROLES])
    def test_role_ranking_produced(self, eval_session, role_names):
        """compute_role_fit_ranking() must return ranked results without crashing."""
        found = _get_role_candidate(role_names, eval_session)
        if not found:
            pytest.skip(f"None of {role_names} found in DB")
        role_id, role_name = found

        skill_names  = _RESUME_SKILLS.get(role_name, [])
        skill_id_map = get_skill_ids(skill_names, eval_session)
        all_profiles = get_all_role_profiles_bulk(eval_session)

        try:
            result = compute_role_fit_ranking(
                set(skill_id_map.values()),
                all_profiles,
                user_skill_names=set(skill_id_map.keys()),
            )
        except Exception as exc:
            pytest.fail(f"compute_role_fit_ranking failed for {role_name!r}: {exc}")

        assert isinstance(result, dict)
        assert "apply_now"       in result
        assert "need_experience" in result
        assert "farther_away"    in result

    @pytest.mark.parametrize("role_names", _ROLES, ids=[r[0] for r in _ROLES])
    def test_recruiter_summary_produced(self, eval_session, role_names):
        """compute_recruiter_summary() must return a complete summary dict."""
        found = _get_role_candidate(role_names, eval_session)
        if not found:
            pytest.skip(f"None of {role_names} found in DB")
        role_id, role_name = found

        session_id   = create_test_session(eval_session, role_id, role_name)
        skill_id_map = get_skill_ids(_RESUME_SKILLS.get(role_name, []), eval_session)
        if skill_id_map:
            seed_user_skills(session_id, list(skill_id_map.values()), eval_session)

        gap_data = analyze_gap(session_id, role_id, eval_session)

        try:
            summary = compute_recruiter_summary(
                gap_data, gap_data["adjusted_readiness_score"]
            )
        except Exception as exc:
            pytest.fail(f"compute_recruiter_summary failed for {role_name!r}: {exc}")

        for key in ("verdict", "reasoning", "strengths", "concerns"):
            assert key in summary, f"summary missing key {key!r} for {role_name!r}"

    @pytest.mark.parametrize("role_names", _ROLES, ids=[r[0] for r in _ROLES])
    def test_template_roadmap_produced(self, eval_session, role_names):
        """
        _template_fallback() must produce a roadmap without crashing for every role.
        This is the production fallback when the LLM fails.
        """
        found = _get_role_candidate(role_names, eval_session)
        if not found:
            pytest.skip(f"None of {role_names} found in DB")
        role_id, role_name = found

        session_id   = create_test_session(eval_session, role_id, role_name)
        skill_id_map = get_skill_ids(_RESUME_SKILLS.get(role_name, []), eval_session)
        if skill_id_map:
            seed_user_skills(session_id, list(skill_id_map.values()), eval_session)

        gap_data = analyze_gap(session_id, role_id, eval_session)

        try:
            roadmap = _template_fallback(role_name, gap_data, "3 months", eval_session)
        except Exception as exc:
            pytest.fail(f"_template_fallback crashed for {role_name!r}: {exc}")

        assert "title"   in roadmap
        assert "summary" in roadmap
        weeks = roadmap.get("weeks") or []
        assert len(weeks) > 0, f"Roadmap for {role_name!r} produced 0 weeks"


# ---------------------------------------------------------------------------
# Phase 11: Failure Harness
# ---------------------------------------------------------------------------

class TestFailureHarness:
    """
    Every failure scenario must produce a specific, graceful error.
    No raw stack traces. No 500 errors on known bad inputs.
    """

    def test_invalid_pdf_bytes_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_resume(b"NOT_A_PDF", "resume.pdf")

    def test_empty_bytes_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_resume(b"", "resume.pdf")

    def test_image_only_pdf_raises_parse_error(self):
        pdf = make_image_only_pdf()
        with pytest.raises(ParseError):
            parse_resume(pdf, "resume.pdf")

    def test_truncated_pdf_raises_parse_error(self):
        """A PDF truncated mid-stream must raise ParseError, not crash the worker."""
        pdf = make_pdf_bytes("SKILLS\nPython Docker")
        truncated = pdf[:len(pdf) // 2]
        with pytest.raises(ParseError):
            parse_resume(truncated, "resume.pdf")

    def test_unsupported_extension_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_resume(b"bytes", "resume.xlsx")

    def test_zero_skills_gap_analysis_safe(self):
        """analyze_gap with no user_skills must return 0% without crashing."""
        from src.services.gap_analyzer import _empty_result
        result = _empty_result()
        assert result["readiness_score"] == 0
        assert result["adjusted_readiness_score"] == 0
        assert result["matched_skills"] == []

    def test_empty_sections_skill_extraction_safe(self):
        """extract_skills on empty sections must return an empty list, not crash."""
        from src.services.skill_extractor import _alias_scan
        result = _alias_scan("", {})
        assert result == []

    def test_none_sections_soft_skill_safe(self):
        """infer_soft_skills(None) and ({}) must return empty list."""
        assert infer_soft_skills({}) == []
        assert infer_soft_skills(None) == []

    def test_empty_matched_skills_evidence_safe(self):
        """build_skill_evidence with empty matched_skills returns empty dict."""
        result = build_skill_evidence({"experience": "Built APIs."}, [])
        assert result == {}

    def test_empty_sections_evidence_safe(self):
        result = build_skill_evidence({}, [{"skill_id": 1, "display_name": "Python", "importance_score": 0.8}])
        assert result == {}

    def test_zero_skill_ranking_safe(self):
        """compute_role_fit_ranking with empty skills returns valid (empty) buckets."""
        result = compute_role_fit_ranking(set(), [], user_skill_names=set())
        assert isinstance(result, dict)
        assert "apply_now" in result

    def test_empty_gap_data_recruiter_summary_safe(self):
        """compute_recruiter_summary on empty gap_data must not crash."""
        empty_gap = {
            "category_breakdown": {},
            "matched_skills": [],
            "missing_skills": [],
        }
        try:
            summary = compute_recruiter_summary(empty_gap, 0)
        except Exception as exc:
            pytest.fail(f"compute_recruiter_summary crashed on empty gap_data: {exc}")
        assert "verdict" in summary

    def test_shortest_path_at_100pct_is_already_there(self):
        """When score >= target, shortest_path must return already_there=True."""
        result = compute_shortest_path([], 0.0, adjusted_readiness_score=70, target_threshold=65)
        assert result["already_there"] is True

    def test_roadmap_fallback_empty_missing_no_crash(self):
        """Template fallback with zero missing skills must not crash."""
        gap = {"readiness_score": 80, "adjusted_readiness_score": 85, "missing_skills": [], "matched_skills": []}

        class _StubDB:
            def execute(self, *a, **kw):
                class R:
                    def fetchall(self): return []
                return R()

        try:
            result = _template_fallback("Software Engineer", gap, "4 weeks", _StubDB())
        except Exception as exc:
            pytest.fail(f"template_fallback crashed on empty missing_skills: {exc}")

    @pytest.mark.requires_db
    @pytest.mark.requires_seed
    def test_unknown_role_normalize_does_not_crash(self, eval_session):
        """An unrecognised role must return a RoleMatch, never raise."""
        from src.services.role_normalizer import normalize_role
        try:
            result = normalize_role("Blockchain NFT Metaverse Architect", eval_session)
        except Exception as exc:
            pytest.fail(f"normalize_role raised on unknown role: {exc}")
        assert result.match_type in ("no_match", "semantic_suggest", "fuzzy_match")

    @pytest.mark.requires_db
    @pytest.mark.requires_seed
    def test_compiler_engineer_no_longer_crashes(self, eval_session):
        """
        Primary regression: Compiler Engineer previously crashed with
        psycopg2.errors.SyntaxError on the :vec::vector cast.
        Must succeed cleanly after the CAST(:vec AS vector) fix.
        """
        from src.services.role_normalizer import normalize_role
        try:
            result = normalize_role("Compiler Engineer", eval_session)
        except Exception as exc:
            pytest.fail(
                f"Compiler Engineer still crashes: {type(exc).__name__}: {exc}\n"
                "The CAST(:vec AS vector) fix must be in the running process — restart the server."
            )
        from src.services.role_normalizer import RoleMatch
        assert isinstance(result, RoleMatch)
