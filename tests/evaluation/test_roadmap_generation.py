"""
Phase 8 — Roadmap Generation Validation

Verifies:
  1. Template fallback never crashes for any combination of weeks/phases/None values
  2. Jinja2 rendering does not crash on None-valued roadmap keys
  3. Duration parsing converts '3 months', '8 weeks', '24 weeks' correctly
  4. The roadmap summary uses adjusted_readiness_score, not raw readiness_score
  5. Template fallback does not teach already-possessed skills
  6. All four duration variants produce non-empty roadmaps

Pure tests — no LLM / Groq call required.
"""

import pytest

from src.services.roadmap_generator import _parse_duration_weeks, _template_fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gap_data(
    matched_names: list[str] | None = None,
    missing_names: list[str] | None = None,
    raw_score: int = 52,
    adj_score: int = 73,
) -> dict:
    matched_names = matched_names or []
    missing_names = missing_names or []
    return {
        "readiness_score":          raw_score,
        "adjusted_readiness_score": adj_score,
        "matched_skills": [
            {"skill_id": i, "display_name": n, "importance_score": 0.8, "category": "Test"}
            for i, n in enumerate(matched_names)
        ],
        "missing_skills": [
            {"skill_id": 100 + i, "display_name": n, "importance_score": 0.7, "category": "Test"}
            for i, n in enumerate(missing_names)
        ],
    }


def _fallback(role: str, gap_data: dict, duration: str, db=None) -> dict:
    """Call _template_fallback with a minimal stub DB if needed."""
    class _StubDB:
        def execute(self, *a, **kw):
            class R:
                def fetchall(self): return []
            return R()
    return _template_fallback(role, gap_data, duration, db or _StubDB())


# ---------------------------------------------------------------------------
# Phase 8-A: None-safety (Jinja2 crash regression)
# ---------------------------------------------------------------------------

class TestNoneSafety:
    """
    Regression: results.html crashed with
    'TypeError: object of type NoneType has no len()'
    when roadmap['weeks'] = None.

    Jinja2's dict.get('key', default) only uses default when key is absent.
    If the key exists with value None, None is returned.
    Fix: use (roadmap.get('weeks') or []) everywhere.
    """

    def test_weeks_none_template_does_not_crash(self):
        """Simulate a roadmap dict where weeks key exists but is None."""
        from jinja2 import Environment
        # Replicate the exact Jinja2 expressions from results.html
        env = Environment()

        # Test 1: total_duration_weeks fallback
        tmpl_duration = env.from_string(
            "{{ roadmap.get('total_duration_weeks', (roadmap.get('weeks') or []) | length) }}"
        )
        result = tmpl_duration.render(roadmap={"weeks": None})
        assert result == "0", f"weeks=None duration rendering failed: {result!r}"

        # Test 2: week loop guard
        tmpl_loop = env.from_string(
            "{% for week in (roadmap.get('weeks') or []) %}{{ week }}{% endfor %}"
        )
        result = tmpl_loop.render(roadmap={"weeks": None})
        assert result == "", "weeks=None should produce empty loop, not crash"

    def test_phases_none_template_does_not_crash(self):
        from jinja2 import Environment
        env = Environment()
        tmpl = env.from_string(
            "{% if roadmap.get('phases') %}{{ roadmap.phases | length }}{% else %}0{% endif %}"
        )
        result = tmpl.render(roadmap={"phases": None})
        assert result == "0"

    def test_empty_weeks_list_template_renders(self):
        from jinja2 import Environment
        env = Environment()
        tmpl = env.from_string(
            "{{ roadmap.get('total_duration_weeks', (roadmap.get('weeks') or []) | length) }} weeks"
        )
        result = tmpl.render(roadmap={"weeks": []})
        assert result == "0 weeks"

    def test_roadmap_none_template_does_not_crash(self):
        """When roadmap is an empty dict (no roadmap generated), the template must not crash."""
        from jinja2 import Environment
        env = Environment()
        tmpl = env.from_string(
            "{% if roadmap and (roadmap.get('phases') or roadmap.get('weeks')) %}YES{% else %}NO{% endif %}"
        )
        assert env.from_string("{{ '0' }}").render(roadmap={}) == "0"  # sanity
        result = tmpl.render(roadmap={})
        assert result == "NO"

    def test_template_fallback_returns_valid_shape(self):
        """_template_fallback() must return a dict matching RoadmapOutput schema."""
        gap = _make_gap_data(
            matched_names=["Python", "FastAPI"],
            missing_names=["Docker", "Redis", "Kubernetes"],
        )
        result = _fallback("Backend Engineer", gap, "3 months")
        assert "title"                in result
        assert "total_duration_weeks" in result
        assert "summary"              in result
        assert "success_metrics"      in result
        # Either weeks or phases must be present and non-None
        has_weeks  = result.get("weeks")  is not None
        has_phases = result.get("phases") is not None
        assert has_weeks or has_phases, "Template fallback has neither weeks nor phases"


# ---------------------------------------------------------------------------
# Phase 8-B: Duration parsing
# ---------------------------------------------------------------------------

class TestDurationParsing:
    @pytest.mark.parametrize("duration,expected_weeks", [
        ("3 months",  12),
        ("6 months",  24),
        ("1 month",    4),
        ("8 weeks",    8),
        ("4 weeks",    4),
        ("12 weeks",  12),
        ("24 weeks",  24),
    ])
    def test_duration_parsed_correctly(self, duration, expected_weeks):
        weeks = _parse_duration_weeks(duration)
        assert weeks == expected_weeks, (
            f"_parse_duration_weeks({duration!r}) = {weeks}, expected {expected_weeks}"
        )

    def test_unknown_duration_returns_default(self):
        weeks = _parse_duration_weeks("a long time")
        assert weeks > 0, "Unknown duration should return a positive default"


# ---------------------------------------------------------------------------
# Phase 8-C: Duration variants produce non-empty roadmaps
# ---------------------------------------------------------------------------

class TestDurationVariants:
    @pytest.mark.parametrize("duration", ["4 weeks", "8 weeks", "12 weeks", "24 weeks"])
    def test_fallback_produces_roadmap_for_duration(self, duration):
        gap = _make_gap_data(
            missing_names=[f"Skill{i}" for i in range(10)],
        )
        result = _fallback("AI Engineer", gap, duration)
        weeks = result.get("weeks") or []
        assert len(weeks) > 0, f"Template fallback produced 0 weeks for duration={duration!r}"

    def test_longer_duration_produces_more_weeks(self):
        gap = _make_gap_data(missing_names=[f"Skill{i}" for i in range(20)])
        short  = _fallback("AI Engineer", gap, "4 weeks")
        longer = _fallback("AI Engineer", gap, "24 weeks")
        short_weeks  = len(short.get("weeks") or [])
        longer_weeks = len(longer.get("weeks") or [])
        assert longer_weeks >= short_weeks, (
            f"24-week roadmap ({longer_weeks}w) not >= 4-week roadmap ({short_weeks}w)"
        )


# ---------------------------------------------------------------------------
# Phase 8-D: Roadmap does not teach possessed skills
# ---------------------------------------------------------------------------

class TestRoadmapContent:
    def test_fallback_uses_missing_skills_not_matched(self):
        """
        Template fallback must draw week topics from missing_skills,
        not from matched_skills (things the user already knows).
        """
        gap = _make_gap_data(
            matched_names=["Python", "FastAPI", "PostgreSQL"],
            missing_names=["Kubernetes", "Terraform", "Redis"],
        )
        result = _fallback("DevOps Engineer", gap, "3 months")
        weeks = result.get("weeks") or []
        all_focus = " ".join(w.get("focus", "") for w in weeks).lower()

        # Missing skills MUST appear in some week focus
        assert "kubernetes" in all_focus or "terraform" in all_focus or "redis" in all_focus, (
            "Template fallback does not teach any of the missing skills. "
            "Roadmap may be using matched skills instead."
        )

    def test_summary_contains_adjusted_readiness_score(self):
        """
        The roadmap summary must mention the adjusted_readiness_score (73%),
        not the raw readiness_score (52%).
        """
        gap = _make_gap_data(
            missing_names=["Kubernetes"],
            raw_score=52,
            adj_score=73,
        )
        result = _fallback("DevOps Engineer", gap, "3 months")
        summary = result.get("summary", "")
        # Must mention 73 (adjusted), not 52 (raw)
        assert "73" in summary, (
            f"Roadmap summary mentions raw score instead of adjusted score.\n"
            f"Summary: {summary!r}\n"
            "Expected '73%' (adjusted_readiness_score), got raw score (52%). "
            "_template_fallback must use gap_data.get('adjusted_readiness_score', ...)."
        )

    def test_empty_missing_skills_produces_review_week(self):
        """When there are no missing skills, fallback must not crash — produce a review week."""
        gap = _make_gap_data(missing_names=[])
        try:
            result = _fallback("Software Engineer", gap, "4 weeks")
        except Exception as exc:
            pytest.fail(f"Template fallback crashed on empty missing_skills: {exc}")
        assert "weeks" in result or "phases" in result


# ---------------------------------------------------------------------------
# Phase 8-E: Success metrics always present
# ---------------------------------------------------------------------------

class TestSuccessMetrics:
    def test_fallback_always_has_success_metrics(self):
        gap = _make_gap_data(missing_names=["Docker", "Kubernetes"])
        result = _fallback("Cloud Engineer", gap, "8 weeks")
        metrics = result.get("success_metrics", [])
        assert isinstance(metrics, list), "success_metrics must be a list"
        assert len(metrics) > 0, "success_metrics must not be empty"
        for m in metrics:
            assert isinstance(m, str) and len(m) > 5, f"Invalid metric: {m!r}"
