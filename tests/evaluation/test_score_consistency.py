"""
Phase 5 — Score Consistency Audit

Verifies that a single role produces ONE canonical readiness score everywhere.

Observed bug (now fixed):
  Executive Summary  → 73%  (adjusted_readiness_score from analyze_gap)
  Role Ranking       → 62%  (simplified formula, no maturity bonus)
  Roadmap header     → 52%  (raw readiness_score, no adjacency, no maturity)

Tests in this file verify:
  1. The scoring formula in compute_role_fit_ranking() uses the same multipliers
     as GAP_MULTIPLIERS in analyze_gap().
  2. Maturity bonus is applied in both analyze_gap() and compute_role_fit_ranking().
  3. Roadmap prompt uses adjusted_readiness_score, not readiness_score.
  4. Template fallback uses adjusted_readiness_score in its summary text.
  5. pages.py passes adjusted_score everywhere (not raw_score).

Pure tests (no DB required): code inspection + functional checks on mocked data.
DB tests: end-to-end score comparison for a synthetic candidate.
"""

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Phase 5-A: Code structure audit (no DB required)
# ---------------------------------------------------------------------------

class TestCodeStructureAudit:
    """
    Inspect source files to detect duplicate scoring engines.
    Fails immediately if the code structure shows inconsistency.
    """

    def _read(self, *parts: str) -> str:
        return (_ROOT / "src" / Path(*parts)).read_text(encoding="utf-8")

    def test_recruiter_imports_gap_multipliers(self):
        """
        compute_role_fit_ranking() must use GAP_MULTIPLIERS from gap_analyzer,
        not hardcoded floats, so both code paths stay in sync automatically.
        """
        source = self._read("services", "recruiter.py")
        assert "GAP_MULTIPLIERS" in source, (
            "recruiter.py does not import or use GAP_MULTIPLIERS from gap_analyzer. "
            "Hardcoded 0.50/0.30 adjacency will diverge from analyze_gap's 0.60/0.35."
        )

    def test_recruiter_imports_maturity_bonus(self):
        """
        Role fit ranking must apply _maturity_bonus() — same function used by analyze_gap.
        Without it, the ranking score can be 10+ points below the Executive Summary score.
        """
        source = self._read("services", "recruiter.py")
        assert "_maturity_bonus" in source, (
            "recruiter.py does not call _maturity_bonus(). "
            "The role ranking score will be consistently lower than the Executive Summary score."
        )

    def test_recruiter_no_hardcoded_050_adjacency(self):
        """
        Hardcoded 0.50 for adjacent_expertise was the bug.
        Now GAP_MULTIPLIERS['adjacent_expertise'] (= 0.60) must be used.
        """
        source = self._read("services", "recruiter.py")
        # Look for the pattern:  v["imp"] * 0.50  or  imp * 0.50  (hardcoded)
        bad = re.search(r'v\["imp"\]\s*\*\s*0\.50', source)
        assert bad is None, (
            "recruiter.py still contains hardcoded 0.50 adjacency credit. "
            "Use GAP_MULTIPLIERS['adjacent_expertise'] instead."
        )

    def test_recruiter_no_hardcoded_030_transferable(self):
        source = self._read("services", "recruiter.py")
        bad = re.search(r'v\["imp"\]\s*\*\s*0\.30', source)
        assert bad is None, (
            "recruiter.py still contains hardcoded 0.30 transferable credit. "
            "Use GAP_MULTIPLIERS['transferable'] instead."
        )

    def test_roadmap_prompt_uses_adjusted_score(self):
        """
        _build_prompt() must inject adjusted_readiness_score into the LLM prompt,
        not readiness_score (the raw number).  This ensures the roadmap text says
        the same score the Executive Summary displays.
        """
        source = self._read("services", "roadmap_generator.py")
        # The line must contain adjusted_readiness_score
        assert "adjusted_readiness_score" in source, (
            "roadmap_generator.py does not reference adjusted_readiness_score. "
            "The roadmap prompt will inject the raw score, causing the '52% vs 73%' mismatch."
        )

    def test_template_fallback_uses_adjusted_score(self):
        """_template_fallback() summary text must reference adjusted_readiness_score."""
        source = self._read("services", "roadmap_generator.py")
        # Find the _template_fallback function and check it uses adjusted score
        # Look for the summary f-string that embeds the score
        assert source.count("adjusted_readiness_score") >= 2, (
            "roadmap_generator.py references adjusted_readiness_score fewer than twice. "
            "One occurrence must be in _build_prompt(), one in _template_fallback()."
        )

    def test_pages_uses_adjusted_score_for_role_ranking(self):
        """pages.py must pass user_skill_names to compute_role_fit_ranking."""
        source = self._read("api", "pages.py")
        assert "user_skill_names" in source, (
            "pages.py does not pass user_skill_names to compute_role_fit_ranking(). "
            "The adjacency fix in recruiter.py is unreachable without this."
        )

    def test_pages_uses_adjusted_score_as_canonical(self):
        """pages.py must read adjusted_readiness_score for the main display score."""
        source = self._read("api", "pages.py")
        assert "adjusted_readiness_score" in source, (
            "pages.py does not extract adjusted_readiness_score from gap_data. "
            "All scores will be the raw keyword-match number."
        )

    def test_session_stores_raw_score_not_adjusted(self):
        """
        analyze_resume.py must store gap_data['readiness_score'] (raw) into the
        session — NOT adjusted — so that pages.py can compute the delta for display.
        The adjusted score is always recomputed fresh from analyze_gap() at render time.
        """
        source = self._read("tasks", "analyze_resume.py")
        # Must reference readiness_score (not adjusted) for the session field
        assert 'session.readiness_score = gap_data["readiness_score"]' in source, (
            "analyze_resume.py does not store the raw readiness_score in session. "
            "The delta display ('adjusted for adjacent expertise') will be broken."
        )


# ---------------------------------------------------------------------------
# Phase 5-B: Functional consistency (pure, no DB)
# ---------------------------------------------------------------------------

class TestFunctionalConsistency:
    """
    End-to-end computation test with synthetic data.
    No DB — constructs gap_data and role_profiles manually.
    """

    def _make_matched_skill(self, name: str, importance: float) -> dict:
        return {
            "skill_id":       hash(name) % 10000,
            "display_name":   name,
            "category":       "Engineering",
            "importance_score": importance,
            "gap_status":     "strong_match",
            "via_skill":      None,
            "confidence":     0.95,
        }

    def _make_gap_data(
        self,
        matched_names: list[str],
        missing_names: list[str],
        raw_score: int = 52,
        adj_score: int = 73,
    ) -> dict:
        return {
            "readiness_score":          raw_score,
            "adjusted_readiness_score": adj_score,
            "matched_skills": [self._make_matched_skill(n, 0.8) for n in matched_names],
            "missing_skills": [
                {
                    "skill_id": hash(n) % 10000 + 50000,
                    "display_name": n,
                    "category": "Engineering",
                    "importance_score": 0.7,
                    "gap_status": "missing",
                    "gap_type": "tooling",
                    "via_skill": None,
                }
                for n in missing_names
            ],
            "bonus_skills": [],
            "category_breakdown": {},
            "score_contributors": [],
            "total_importance": 5.0,
            "user_skill_ids": {},
        }

    def test_recruiter_summary_uses_passed_score(self):
        """compute_recruiter_summary takes readiness_score as an argument — no internal re-scoring."""
        from src.services.recruiter import compute_recruiter_summary
        gap_data = self._make_gap_data(["Python", "FastAPI"], ["Docker"])

        summary_73 = compute_recruiter_summary(gap_data, 73)
        summary_52 = compute_recruiter_summary(gap_data, 52)

        assert summary_73["verdict"] != summary_52["verdict"] or summary_73["reasoning"] != summary_52["reasoning"], (
            "compute_recruiter_summary ignores the readiness_score argument — it returns the same result for 73 and 52. "
            "This means the recruiter verdict is wrong."
        )

    def test_gap_multipliers_match_between_modules(self):
        """GAP_MULTIPLIERS in gap_analyzer and recruiter must be the same object."""
        from src.services.gap_analyzer import GAP_MULTIPLIERS as gm_ga
        from src.services.recruiter import GAP_MULTIPLIERS as gm_rec
        assert gm_ga is gm_rec or gm_ga == gm_rec, (
            "GAP_MULTIPLIERS in recruiter.py is not the same as in gap_analyzer.py. "
            "One of the imports is stale or there's a local override."
        )

    def test_adjacent_expertise_multiplier_is_060(self):
        """adjacent_expertise must be 0.60, not 0.50."""
        from src.services.gap_analyzer import GAP_MULTIPLIERS
        val = GAP_MULTIPLIERS.get("adjacent_expertise")
        assert val == 0.60, (
            f"GAP_MULTIPLIERS['adjacent_expertise'] = {val}, expected 0.60. "
            "The role ranking will use this value. If it's 0.50, ranking will be ~10pts low."
        )

    def test_transferable_multiplier_is_035(self):
        """transferable must be 0.35, not 0.30."""
        from src.services.gap_analyzer import GAP_MULTIPLIERS
        val = GAP_MULTIPLIERS.get("transferable")
        assert val == 0.35, (
            f"GAP_MULTIPLIERS['transferable'] = {val}, expected 0.35."
        )

    def test_maturity_bonus_applied_consistently(self):
        """
        _maturity_bonus() with FastAPI + Python + REST API + Machine Learning + Deep Learning + Git
        should return 10 (capped) in both analyze_gap and recruiter contexts.
        """
        from src.services.gap_analyzer import _maturity_bonus, _MATURITY_BONUS_CAP
        high_signal_skills = [
            {"display_name": "FastAPI"},       # +2
            {"display_name": "Python"},        # +1
            {"display_name": "REST API"},      # +2
            {"display_name": "Machine Learning"}, # +2
            {"display_name": "Deep Learning"}, # +2
            {"display_name": "Git"},           # +1
        ]
        bonus = _maturity_bonus(high_signal_skills)
        # Total would be 10, capped at _MATURITY_BONUS_CAP
        assert bonus == _MATURITY_BONUS_CAP, (
            f"_maturity_bonus returned {bonus}, expected {_MATURITY_BONUS_CAP}. "
            "High-signal ML/backend resume should hit the maturity cap."
        )


# ---------------------------------------------------------------------------
# Phase 5-C: DB-backed end-to-end score consistency
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestEndToEndScoreConsistency:
    """
    Verify that for the same candidate, the adjusted score from analyze_gap()
    is within ±5 pts of the ranking score from compute_role_fit_ranking().
    Larger divergence means the formulas are still inconsistent.
    """

    def test_ranking_score_within_5pts_of_analyze_gap(self, eval_session):
        from src.services.gap_analyzer import analyze_gap, get_all_role_profiles_bulk
        from src.services.recruiter import compute_role_fit_ranking
        from .helpers import get_role_id, get_skill_ids, create_test_session, seed_user_skills

        skill_names = [
            "Python", "FastAPI", "REST API", "PostgreSQL", "Docker",
            "Machine Learning", "Deep Learning", "Git",
        ]
        skill_id_map = get_skill_ids(skill_names, eval_session)
        if len(skill_id_map) < 3:
            pytest.skip("Not enough skills found in DB — run `make seed`")

        # Find a role to test against
        from sqlalchemy import text
        row = eval_session.execute(
            text("SELECT role_id, display_name FROM role_categories LIMIT 1")
        ).first()
        if not row:
            pytest.skip("No roles in DB")

        role_id = row.role_id
        role_name = row.display_name

        # Create test session and seed user skills
        session_id = create_test_session(eval_session, role_id, role_name)
        seed_user_skills(session_id, list(skill_id_map.values()), eval_session)

        # Score via analyze_gap
        gap_data = analyze_gap(session_id, role_id, eval_session)
        adjusted = gap_data["adjusted_readiness_score"]

        # Score via compute_role_fit_ranking
        all_profiles = get_all_role_profiles_bulk(eval_session)
        ranking = compute_role_fit_ranking(
            set(skill_id_map.values()),
            all_profiles,
            current_role_id=None,  # don't exclude the test role
            user_skill_names=set(skill_id_map.keys()),
        )
        all_results = (
            ranking.get("apply_now", [])
            + ranking.get("need_experience", [])
            + ranking.get("farther_away", [])
        )
        rank_score = next(
            (r["fit_pct"] for r in all_results if r["role_id"] == role_id),
            None,
        )
        if rank_score is None:
            pytest.skip(f"Role {role_name!r} not in ranking results")

        diff = abs(adjusted - rank_score)
        assert diff <= 5, (
            f"Score inconsistency for {role_name!r}: "
            f"analyze_gap={adjusted}% vs role_ranking={rank_score}% (diff={diff}pts). "
            f"Should be within 5pts — larger difference means duplicate scoring engines."
        )
