"""
Phase 7 — Soft Skill Inference Validation

Verifies that infer_soft_skills() has precision over optimism:

  MUST infer:
    "Led a team of 5 engineers"          → Leadership
    "Mentored junior engineers"          → Mentorship (user is the mentor)
    "Collaborated with cross-functional" → Collaboration
    "Reduced latency by 40%"             → Impact Orientation

  MUST NOT infer:
    "Collaborated with mentors"          → NOT Leadership
    "Collaborated with mentors"          → NOT Mentorship (user is the mentee)
    "Worked with a team"                 → NOT Leadership (not a leadership verb)
    A very short match                   → NOT above 0.65 confidence floor

  Confidence floor:
    Any inference below 0.65 must be dropped.
    Results with no evidence snippets must be dropped.

All tests are pure function tests — no DB required.
"""

import pytest

from src.services.soft_skill_inferencer import infer_soft_skills


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inferred_skills(text: str) -> dict[str, dict]:
    """Run infer_soft_skills on a single-section resume. Returns {skill: entry}."""
    sections = {"experience": text}
    results = infer_soft_skills(sections)
    return {r["skill"]: r for r in results}


def _assert_not_inferred(inferences: dict, skill: str, context: str):
    if skill in inferences:
        pytest.fail(
            f"'{skill}' incorrectly inferred from: {context!r}\n"
            f"Evidence: {inferences[skill].get('evidence')}\n"
            f"Confidence: {inferences[skill].get('confidence')}"
        )


# ---------------------------------------------------------------------------
# Phase 7-A: Should infer — true positives
# ---------------------------------------------------------------------------

class TestTruePositives:
    def test_led_a_team_infers_leadership(self):
        inf = _inferred_skills("Led a team of 5 engineers to deliver the new authentication system.")
        assert "Leadership" in inf, (
            "'Led a team of 5 engineers' must infer Leadership. "
            "Check leadership patterns in _INFERENCE_RULES."
        )

    def test_managed_a_team_infers_leadership(self):
        inf = _inferred_skills("Managed a team of 3 engineers during the Q3 platform migration.")
        assert "Leadership" in inf, (
            "'Managed a team' must infer Leadership."
        )

    def test_spearheaded_infers_leadership(self):
        inf = _inferred_skills("Spearheaded the migration of our monolith to microservices architecture.")
        assert "Leadership" in inf

    def test_team_lead_title_infers_leadership(self):
        inf = _inferred_skills("Served as team lead for the backend infrastructure squad.")
        assert "Leadership" in inf

    def test_mentored_junior_engineers_infers_mentorship(self):
        """
        Regression: 'mentored junior engineers' must infer Mentorship.
        The user is the mentor (active past tense + target is junior/intern).
        """
        inf = _inferred_skills("Mentored junior engineers on Python best practices and code reviews.")
        assert "Mentorship" in inf, (
            "'Mentored junior engineers' must infer Mentorship. "
            "User is the mentor, not the mentee."
        )

    def test_coached_intern_infers_mentorship(self):
        inf = _inferred_skills("Coached intern developers on FastAPI design patterns.")
        assert "Mentorship" in inf

    def test_collaborated_with_team_infers_collaboration(self):
        inf = _inferred_skills("Collaborated with a cross-functional team of 10 engineers and designers.")
        assert "Collaboration" in inf or "Teamwork" in inf, (
            "'Collaborated with' must infer Collaboration or Teamwork."
        )

    def test_reduced_by_percentage_infers_impact(self):
        inf = _inferred_skills("Reduced database query latency by 40% using query optimization.")
        assert "Impact Orientation" in inf or "Problem Solving" in inf, (
            "'Reduced ... by 40%' must infer Impact Orientation or Problem Solving."
        )

    def test_open_source_contribution_infers_oss(self):
        inf = _inferred_skills("Contributed to open-source FastAPI middleware; PR merged upstream.")
        assert "Open Source Contribution" in inf


# ---------------------------------------------------------------------------
# Phase 7-B: Should NOT infer — false positive prevention
# ---------------------------------------------------------------------------

class TestFalsePositives:
    def test_collaborated_with_mentors_not_leadership(self):
        """
        Regression: 'Collaborated with mentors' must NOT infer Leadership.
        The user is working WITH mentors (i.e. they are the mentee).
        """
        inf = _inferred_skills("Collaborated with Samsung R&D mentors on ETL pipeline research.")
        _assert_not_inferred(inf, "Leadership",
            "Collaborated with Samsung R&D mentors on ETL pipeline research.")

    def test_collaborated_with_mentors_not_mentorship(self):
        """
        Regression: 'Collaborated with mentors' must NOT infer Mentorship.
        The mentors here are mentoring the user — the user is not mentoring anyone.
        """
        inf = _inferred_skills("Collaborated with Samsung R&D mentors on ETL pipeline research.")
        _assert_not_inferred(inf, "Mentorship",
            "Collaborated with Samsung R&D mentors on ETL pipeline research.")

    def test_worked_with_a_team_not_leadership(self):
        """
        Passive team participation is not leadership. No leadership verb present.
        """
        inf = _inferred_skills("Worked alongside a team of 5 engineers on the platform.")
        _assert_not_inferred(inf, "Leadership",
            "Worked alongside a team of 5 engineers.")

    def test_worked_with_mentors_not_mentorship(self):
        """
        'Worked with mentors' — user is being mentored, not mentoring.
        """
        inf = _inferred_skills("Worked with mentors from the research lab on my final year project.")
        _assert_not_inferred(inf, "Mentorship",
            "Worked with mentors from the research lab.")

    def test_received_guidance_from_senior_not_mentorship(self):
        inf = _inferred_skills("Received guidance from senior engineers on system design principles.")
        _assert_not_inferred(inf, "Mentorship",
            "Received guidance from senior engineers.")

    def test_learning_from_mentors_not_leadership(self):
        inf = _inferred_skills("Learning from experienced mentors in the R&D team daily.")
        _assert_not_inferred(inf, "Leadership",
            "Learning from experienced mentors.")

    def test_participated_in_sprint_not_leadership(self):
        """Agile participation alone is not leadership."""
        inf = _inferred_skills("Participated in weekly sprint planning and daily standups.")
        _assert_not_inferred(inf, "Leadership",
            "Participated in weekly sprint planning.")


# ---------------------------------------------------------------------------
# Phase 7-C: Confidence floor
# ---------------------------------------------------------------------------

class TestConfidenceFloor:
    def test_no_result_below_065_confidence(self):
        """
        All returned inferences must be >= 0.65 confidence.
        Results below this threshold are suppressed to prevent false precision.
        """
        # Provide a rich resume so multiple inferences are possible
        text = (
            "Led a team of 3. Mentored junior developers. "
            "Collaborated with cross-functional teams. Open-source contributor. "
            "Reduced latency by 30%. Presented results to stakeholders. "
            "Sprint planning. Self-taught Python. Independently developed ETL pipeline."
        )
        results = infer_soft_skills({"experience": text})
        for r in results:
            assert r["confidence"] >= 0.65, (
                f"Inference {r['skill']!r} has confidence {r['confidence']} below 0.65 floor. "
                "It should have been filtered out."
            )

    def test_no_result_without_evidence_snippet(self):
        """
        Every returned inference must have at least one non-empty evidence snippet.
        An inference without evidence is unsubstantiated.
        """
        text = (
            "Led the backend team. Mentored junior engineers. "
            "Collaborated across functions. Contributed to open-source projects."
        )
        results = infer_soft_skills({"experience": text})
        for r in results:
            assert r.get("evidence") and len(r["evidence"]) > 0, (
                f"Inference {r['skill']!r} has no evidence snippets. "
                f"Confidence={r['confidence']} but evidence={r['evidence']!r}"
            )

    def test_empty_resume_produces_no_inferences(self):
        results = infer_soft_skills({})
        assert results == [], f"Empty sections produced inferences: {results}"

    def test_skills_section_only_produces_no_inferences(self):
        """Bare skill lists have no behavioral language — should infer nothing."""
        results = infer_soft_skills({"skills": "Python, Docker, PostgreSQL, FastAPI, Git"})
        assert results == [], (
            f"Skills-only section produced inferences: {results}"
        )


# ---------------------------------------------------------------------------
# Phase 7-D: Direction detection
# ---------------------------------------------------------------------------

class TestDirectionDetection:
    def test_mentoring_past_tense_active_infers_mentorship(self):
        """'mentored [junior|intern|new|student|team member]' — user is the mentor."""
        for target in ("junior developers", "intern engineers", "new team members", "student interns"):
            inf = _inferred_skills(f"Mentored {target} on software engineering best practices.")
            assert "Mentorship" in inf, (
                f"'Mentored {target}' did not infer Mentorship."
            )

    def test_onboarding_others_infers_mentorship(self):
        inf = _inferred_skills("Onboarded junior engineers to the codebase and development workflow.")
        assert "Mentorship" in inf

    def test_training_junior_infers_mentorship(self):
        inf = _inferred_skills("Trained new engineers on Docker and CI/CD pipelines.")
        assert "Mentorship" in inf

    def test_led_mentorship_program_infers_both(self):
        """Leading a mentorship program should infer both Leadership and Mentorship."""
        inf = _inferred_skills(
            "Led a team of 5 engineers. Mentored junior developers through the onboarding program."
        )
        assert "Leadership" in inf, "Expected Leadership from 'Led a team'"
        assert "Mentorship" in inf, "Expected Mentorship from 'Mentored junior developers'"

    def test_mentored_does_not_infer_leadership_alone(self):
        """
        Mentorship alone must NOT infer Leadership.
        Leadership was previously added to the mentorship inference rule — removed in the fix.
        """
        inf = _inferred_skills("Mentored junior engineers on Python and REST API design.")
        # Mentorship should be inferred
        assert "Mentorship" in inf
        # Leadership must NOT come from mentorship (no leadership verb present)
        if "Leadership" in inf:
            # Check that it has a real leadership-verb snippet, not a mentorship snippet
            leadership_ev = inf["Leadership"].get("evidence", [])
            for snippet in leadership_ev:
                snippet_lower = snippet.lower()
                has_lead_verb = any(v in snippet_lower for v in [
                    "led ", "managed ", "team lead", "tech lead", "spearheaded",
                    "owned the", "supervised", "directed",
                ])
                assert has_lead_verb, (
                    f"Leadership inferred without a leadership verb. "
                    f"Evidence came from mentorship context: {snippet!r}. "
                    "Leadership must not be inferred from mentorship patterns alone."
                )
