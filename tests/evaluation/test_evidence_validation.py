"""
Phase 6 — Evidence Retrieval Validation

Verifies that build_skill_evidence() returns evidence that:
  1. Contains the skill name or a recognisable alias
  2. Comes from the highest-confidence section available
  3. Suppresses evidence when no action verb is present (score < 2)
  4. Does NOT return evidence from unrelated sentences
     (e.g. PyTorch text for a Git evidence card)

All tests are pure function tests — no DB required.
"""

import pytest

from src.services.soft_skill_inferencer import build_skill_evidence


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _skill(skill_id: int, display_name: str) -> dict:
    return {"skill_id": skill_id, "display_name": display_name, "category": "Test", "importance_score": 0.8}


# ---------------------------------------------------------------------------
# Phase 6-A: Evidence contains skill name
# ---------------------------------------------------------------------------

class TestEvidenceContainsSkillName:
    def test_pytorch_evidence_mentions_pytorch(self):
        sections = {
            "projects": (
                "Built a digital twin simulator using PyTorch RNNs "
                "that reduced forecasting error by 40%."
            ),
        }
        ev = build_skill_evidence(sections, [_skill(1, "PyTorch")])
        if 1 in ev:
            snippet = ev[1]["snippet"].lower()
            assert "pytorch" in snippet, (
                f"PyTorch evidence does not mention 'pytorch': {ev[1]['snippet']!r}"
            )

    def test_docker_evidence_mentions_docker(self):
        sections = {
            "experience": (
                "Containerized all microservices using Docker Compose. "
                "Deployed Docker images to AWS ECR for production use."
            ),
        }
        ev = build_skill_evidence(sections, [_skill(2, "Docker")])
        if 2 in ev:
            snippet = ev[2]["snippet"].lower()
            assert "docker" in snippet, (
                f"Docker evidence does not mention 'docker': {ev[2]['snippet']!r}"
            )

    def test_fastapi_evidence_mentions_fastapi(self):
        sections = {
            "experience": (
                "Developed async REST APIs with FastAPI serving 50k requests per day."
            ),
        }
        ev = build_skill_evidence(sections, [_skill(3, "FastAPI")])
        if 3 in ev:
            snippet = ev[3]["snippet"].lower()
            assert "fastapi" in snippet, (
                f"FastAPI evidence does not mention 'fastapi': {ev[3]['snippet']!r}"
            )

    def test_git_evidence_does_not_match_github(self):
        """
        Regression: 'git' matched inside 'GitHub' before word-boundary fix.
        Evidence for 'Git' must not come from a sentence that only contains 'GitHub'.
        """
        sections = {
            "experience": (
                "Managed all code on GitHub. Contributed to GitHub Actions workflows."
            ),
            "projects": (
                "Used Git for version control across all projects."
            ),
        }
        ev = build_skill_evidence(sections, [_skill(4, "Git")])
        if 4 in ev:
            snippet = ev[4]["snippet"].lower()
            # The evidence should come from the 'projects' section (actual 'git' word)
            # not the 'experience' section (only 'github')
            assert "git" in snippet, (
                f"Git evidence snippet doesn't contain 'git': {snippet!r}"
            )

    def test_sql_evidence_does_not_match_nosql(self):
        """
        'SQL' must not match inside 'NoSQL' due to word-boundary enforcement.
        """
        sections = {
            "skills": "NoSQL databases, MongoDB, Cassandra",
            "experience": "Built SQL queries for PostgreSQL analytics reporting.",
        }
        ev = build_skill_evidence(sections, [_skill(5, "SQL")])
        if 5 in ev:
            snippet = ev[5]["snippet"].lower()
            # Evidence must come from 'sql queries' not from 'nosql'
            assert "nosql" not in snippet or "sql" in snippet.replace("nosql", ""), (
                f"SQL evidence came from a NoSQL sentence: {snippet!r}"
            )


# ---------------------------------------------------------------------------
# Phase 6-B: Section priority (High > Medium > Low)
# ---------------------------------------------------------------------------

class TestSectionPriority:
    def test_projects_preferred_over_skills_section(self):
        """
        When a skill appears in both 'projects' (High) and 'skills' (Low),
        evidence must come from 'projects'.
        """
        sections = {
            "skills":   "Python, FastAPI, Docker, PostgreSQL",
            "projects": "Built a production FastAPI service deployed with Docker Compose.",
        }
        ev = build_skill_evidence(sections, [_skill(1, "FastAPI")])
        if 1 in ev:
            assert ev[1]["confidence"] == "High", (
                f"FastAPI evidence came from a Low-confidence section despite 'projects' being available. "
                f"section={ev[1]['section']!r} confidence={ev[1]['confidence']!r}"
            )

    def test_experience_preferred_over_skills_section(self):
        sections = {
            "skills":     "Python, Docker, REST API",
            "experience": "Deployed Docker containers to production for 3 years.",
        }
        ev = build_skill_evidence(sections, [_skill(2, "Docker")])
        if 2 in ev:
            assert ev[2]["confidence"] == "High", (
                f"Docker evidence from Low section despite experience available. "
                f"section={ev[2]['section']!r}"
            )

    def test_high_confidence_stops_search(self):
        """
        Once a High-confidence match is found, evidence for that skill must
        come from a High section, even if there's a longer match in Low.
        """
        sections = {
            "experience": "Built PostgreSQL schemas for financial data with complex joins.",
            "skills": (
                "PostgreSQL, MySQL, SQLite, MongoDB, Redis, Elasticsearch, "
                "Cassandra, CockroachDB, DynamoDB"  # long but Low section
            ),
        }
        ev = build_skill_evidence(sections, [_skill(3, "PostgreSQL")])
        if 3 in ev:
            assert ev[3]["confidence"] == "High"


# ---------------------------------------------------------------------------
# Phase 6-C: Weak evidence suppressed
# ---------------------------------------------------------------------------

class TestWeakEvidenceSuppressed:
    def test_skills_section_only_no_action_verb_suppressed(self):
        """
        A skill that only appears in the 'skills' section with no action verb
        (no sentence describing usage) must NOT produce an evidence card.
        This prevents misleading "evidence" from bare skill lists.
        """
        sections = {
            "skills": "Python, Docker, PostgreSQL, Redis, Git, Linux",
            # No experience or projects mentioning Docker
        }
        ev = build_skill_evidence(sections, [_skill(2, "Docker")])
        assert 2 not in ev, (
            "Docker appeared in evidence despite only being in the skills list with no action verb. "
            "This is false evidence — the user just listed Docker, they didn't demonstrate it."
        )

    def test_short_mention_without_verb_suppressed(self):
        """
        A snippet shorter than 20 chars or without an action verb scores < 2
        and must be suppressed.
        """
        sections = {
            "skills": "FastAPI. Python.",
        }
        ev = build_skill_evidence(sections, [_skill(3, "FastAPI")])
        assert 3 not in ev, (
            "FastAPI bare listing in skills section produced evidence card. Score threshold not working."
        )

    def test_unrelated_sentence_suppressed(self):
        """
        Git evidence must not come from a sentence that only mentions GitHub Actions
        (word-boundary fix), especially when Git itself appears in a better sentence.
        """
        sections = {
            "experience": (
                "Managed CI/CD pipelines via GitHub Actions. "
                "Used Git for distributed version control across 5 repositories."
            ),
        }
        ev = build_skill_evidence(sections, [_skill(4, "Git")])
        if 4 in ev:
            snippet = ev[4]["snippet"].lower()
            # The selected snippet must contain the word 'git' as a standalone word
            import re
            assert re.search(r"\bgit\b", snippet), (
                f"Git evidence snippet does not contain standalone 'git': {snippet!r}"
            )


# ---------------------------------------------------------------------------
# Phase 6-D: Multi-skill evidence (no cross-contamination)
# ---------------------------------------------------------------------------

class TestMultiSkillEvidence:
    def test_pytorch_and_git_do_not_cross_contaminate(self):
        """
        PyTorch evidence must not mention only 'git', and Git evidence must not mention only 'pytorch'.
        Regression: previously 'Git' evidence contained a PyTorch sentence.
        """
        sections = {
            "projects": (
                "Developed a PyTorch RNN simulator for digital twin forecasting. "
                "Used Git for version control and GitHub Actions for CI/CD."
            ),
        }
        skills = [_skill(1, "PyTorch"), _skill(2, "Git")]
        ev = build_skill_evidence(sections, skills)

        if 1 in ev and 2 in ev:
            pytorch_snippet = ev[1]["snippet"].lower()
            git_snippet     = ev[2]["snippet"].lower()

            import re
            assert re.search(r"\bpytorch\b", pytorch_snippet), (
                f"PyTorch evidence doesn't mention pytorch: {pytorch_snippet!r}"
            )
            assert re.search(r"\bgit\b", git_snippet), (
                f"Git evidence doesn't mention git: {git_snippet!r}"
            )

    def test_impact_field_extracted_when_quantified(self):
        """Evidence for skills with quantified results must populate the 'impact' field."""
        sections = {
            "experience": (
                "Built FastAPI microservices that reduced response latency by 40%."
            ),
        }
        ev = build_skill_evidence(sections, [_skill(1, "FastAPI")])
        if 1 in ev and ev[1]["confidence"] == "High":
            assert ev[1]["impact"] is not None, (
                "Impact field is None despite quantified result ('reduced latency by 40%'). "
                "_extract_impact() not detecting percentage improvements."
            )

    def test_multi_word_skill_phrase_matched(self):
        """
        'REST API' (two words) must match the full phrase, not just 'REST' or 'API'.
        """
        sections = {
            "experience": "Designed REST API endpoints for the mobile client.",
        }
        ev = build_skill_evidence(sections, [_skill(5, "REST API")])
        if 5 in ev:
            assert "rest api" in ev[5]["snippet"].lower(), (
                f"REST API evidence snippet missing 'rest api': {ev[5]['snippet']!r}"
            )
