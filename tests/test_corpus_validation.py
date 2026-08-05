"""
Skill Corpus Validation Test Suite
====================================
Runs all benchmark tiers as pytest assertions.

Test classes:
  TestBoundaryPattern      — _make_boundary_pattern unit tests (9 cases)
  TestNormalizationProbes  — Tier 1: all probes must pass (T1a / T1b / T1c / T1d)
  TestResumePhraseRecall   — Tier 2: overall recall >= 0.90, no phrase gets 0%
  TestAdversarialFP        — Tier 3: critical-severity probes must be clean
  TestAliasCoverage        — Sanity checks on skills_master.csv shape
  TestAliasCollisions      — No alias shared by two different canonicals
  TestSeedIntegrity        — seed_db.py ROLE_SKILL_PROFILES has no broken refs

Run all:
    pytest tests/test_corpus_validation.py -v

Run only fast unit tests (no CSV loading):
    pytest tests/test_corpus_validation.py::TestBoundaryPattern -v
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_corpus import (
    _make_boundary_pattern,
    build_alias_map,
    check_m1,
    check_m6,
    load_skills,
    scan_text,
)

DATA_DIR = ROOT / "data"
CORPUS_DIR = ROOT / "tests" / "corpus"


# ─────────────────── Session fixtures ────────────────────────────────────────

@pytest.fixture(scope="session")
def skills():
    return load_skills(DATA_DIR)


@pytest.fixture(scope="session")
def alias_map(skills):
    am, _ = build_alias_map(skills)
    return am


@pytest.fixture(scope="session")
def tier1_probes():
    path = CORPUS_DIR / "tier1_normalization_probes.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["probes"]


@pytest.fixture(scope="session")
def tier2_phrases():
    path = CORPUS_DIR / "tier2_resume_phrases.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["phrases"]


@pytest.fixture(scope="session")
def tier3_probes():
    path = CORPUS_DIR / "tier3_adversarial.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["probes"]


# ─────────────────── TestBoundaryPattern ─────────────────────────────────────

class TestBoundaryPattern:
    """Unit tests for _make_boundary_pattern — no CSV required."""

    def test_alphanumeric_alias_uses_b_boundaries(self):
        pat = _make_boundary_pattern("python")
        assert pat == r"\bpython\b"

    def test_alias_ending_nonword_uses_negative_lookahead(self):
        pat = _make_boundary_pattern("c++")
        assert pat.endswith(r"(?!\w)"), f"Expected lookahead suffix, got: {pat!r}"

    def test_alias_starting_nonword_uses_negative_lookbehind(self):
        pat = _make_boundary_pattern(".net c#")
        assert pat.startswith(r"(?<!\w)"), f"Expected lookbehind prefix, got: {pat!r}"

    def test_cpp_matches_standalone_in_sentence(self):
        pat = _make_boundary_pattern("c++")
        assert re.search(pat, "i write c++ code"), "c++ should match standalone"

    def test_cpp_does_not_match_inside_c11_version(self):
        # c++ must NOT match the c++ prefix inside "c++11" (c++11 has its own alias)
        pat = _make_boundary_pattern("c++")
        assert not re.search(pat, "experienced in c++11 and c++17"), (
            "c++ must not match inside c++11 — (?!\\w) should block it"
        )

    def test_csharp_alias_matches(self):
        pat = _make_boundary_pattern("c#")
        assert re.search(pat, "c# developer"), "c# should match standalone"

    def test_fsharp_alias_matches(self):
        pat = _make_boundary_pattern("f#")
        assert re.search(pat, "functional programming in f#"), "f# should match"

    def test_dotnet_csharp_compound_alias_matches(self):
        pat = _make_boundary_pattern(".net c#")
        assert re.search(pat, "worked with .net c# and azure"), ".net c# should match"

    def test_word_boundary_blocks_substring_false_positives(self):
        # "go" must NOT match inside "algorithm", "logo", "Django"
        pat = _make_boundary_pattern("go")
        assert not re.search(pat, "algorithm-based logo in django"), (
            "go must not match inside compound words"
        )

    def test_node_js_matches_with_embedded_period(self):
        pat = _make_boundary_pattern("node.js")
        assert re.search(pat, "built with node.js and express"), "node.js should match"


# ─────────────────── TestNormalizationProbes ─────────────────────────────────

class TestNormalizationProbes:
    """
    Tier 1: every probe in tier1_normalization_probes.json must pass.
    A failure here means a canonical alias is missing or the scan function
    has a boundary bug that escaped the unit tests above.
    """

    def test_all_tier1_probes_pass(self, tier1_probes, alias_map):
        failures = []
        for probe in tier1_probes:
            found = scan_text(probe["input"], alias_map)
            if probe["expected"] not in found:
                failures.append(
                    f"  [{probe.get('subtier','?')}] {probe['description']!r}\n"
                    f"    input={probe['input']!r}\n"
                    f"    expected={probe['expected']!r}\n"
                    f"    got={sorted(found)}"
                )
        assert not failures, (
            f"{len(failures)} normalization probe(s) FAILED:\n" + "\n".join(failures)
        )

    def test_t1b_docker_ce_resolves_to_docker(self, alias_map):
        """Explicit check for the user-reported Docker CE → Docker requirement."""
        assert "docker" in scan_text("Docker CE", alias_map), (
            "'Docker CE' must resolve to 'docker' — add alias to skills_master.csv"
        )

    def test_t1b_docker_community_edition_resolves(self, alias_map):
        assert "docker" in scan_text("Docker Community Edition", alias_map)

    def test_t1b_fast_api_spaced_resolves(self, alias_map):
        assert "fastapi" in scan_text("Fast API", alias_map)

    def test_t1b_postgres_short_resolves(self, alias_map):
        assert "postgresql" in scan_text("Postgres", alias_map)

    def test_t1c_cpp_resolves(self, alias_map):
        assert "cpp" in scan_text("C++", alias_map), (
            "C++ must match — _make_boundary_pattern lookahead fix required"
        )

    def test_t1c_csharp_resolves(self, alias_map):
        assert "csharp" in scan_text("C#", alias_map)

    def test_t1c_fsharp_resolves(self, alias_map):
        assert "fsharp" in scan_text("F#", alias_map)

    def test_t1d_nodejs_in_sentence(self, alias_map):
        found = scan_text("Developed applications using Node.js and React", alias_map)
        assert "nodejs" in found, "Node.js alias must resolve to nodejs canonical"


# ─────────────────── TestResumePhraseRecall ──────────────────────────────────

class TestResumePhraseRecall:
    """
    Tier 2: overall recall across all resume phrases must be >= THRESHOLD.
    No individual phrase should have 0% recall.
    """

    RECALL_THRESHOLD = 0.90

    def test_overall_recall_meets_threshold(self, tier2_phrases, alias_map):
        total_expected = 0
        total_matched = 0
        for phrase in tier2_phrases:
            expected = set(phrase["expected_canonicals"])
            got = scan_text(phrase["text"], alias_map)
            total_expected += len(expected)
            total_matched += len(expected & got)

        recall = total_matched / total_expected if total_expected else 1.0
        assert recall >= self.RECALL_THRESHOLD, (
            f"Overall recall {recall:.1%} is below the {self.RECALL_THRESHOLD:.0%} threshold. "
            f"Matched {total_matched}/{total_expected} expected skill extractions."
        )

    def test_no_phrase_has_zero_recall(self, tier2_phrases, alias_map):
        zero_recall = []
        for phrase in tier2_phrases:
            expected = set(phrase["expected_canonicals"])
            if not expected:
                continue
            got = scan_text(phrase["text"], alias_map)
            if not (expected & got):
                zero_recall.append(
                    f"  {phrase['description']!r}: missed all of {sorted(expected)}"
                )
        assert not zero_recall, (
            f"{len(zero_recall)} phrase(s) with 0% recall:\n" + "\n".join(zero_recall)
        )


# ─────────────────── TestAdversarialFP ───────────────────────────────────────

class TestAdversarialFP:
    """
    Tier 3: critical-severity probes must produce zero forbidden extractions.
    Known-risk probes (English-word FP) are NOT tested here — they're documented
    in the validate_corpus.py report as tracked limitations.
    """

    def test_no_critical_probe_triggers_forbidden_canonical(self, tier3_probes, alias_map):
        failures = []
        for probe in tier3_probes:
            if probe.get("severity") != "critical":
                continue
            forbidden = set(probe.get("forbidden_canonicals", []))
            if not forbidden:
                continue
            extracted = scan_text(probe["text"], alias_map)
            violations = extracted & forbidden
            if violations:
                failures.append(
                    f"  {probe['description']!r}\n"
                    f"    text={probe['text'][:80]!r}\n"
                    f"    forbidden={sorted(forbidden)}\n"
                    f"    extracted={sorted(extracted)}\n"
                    f"    violations={sorted(violations)}"
                )
        assert not failures, (
            f"{len(failures)} critical adversarial probe(s) triggered forbidden extractions:\n"
            + "\n".join(failures)
        )

    def test_go_does_not_match_algorithm(self, alias_map):
        """Regression: \b fix must prevent 'go' from matching inside 'algorithm'."""
        found = scan_text("The algorithm runs in O(n log n)", alias_map)
        assert "go" not in found, "'go' matched inside 'algorithm' or 'log' — boundary fix broken"

    def test_go_does_not_match_logo_or_django(self, alias_map):
        """Regression: 'go' must not match inside 'logo' or 'Django'."""
        found = scan_text("The Django framework has a distinctive logo", alias_map)
        assert "go" not in found, "'go' matched inside 'logo' or 'Django'"

    @pytest.mark.xfail(
        reason="'salt' alias for SaltStack is a known English-word FP — documented as known_risk in tier3_adversarial.json",
        strict=False,
    )
    def test_saltstack_does_not_match_culinary_salt(self, alias_map):
        found = scan_text("Add salt and pepper to taste", alias_map)
        assert "saltstack" not in found


# ─────────────────── TestAliasCoverage ───────────────────────────────────────

class TestAliasCoverage:
    """Sanity checks on skills_master.csv structure."""

    def test_total_skill_count(self, skills):
        assert len(skills) >= 500, f"Expected >= 500 skills, got {len(skills)}"

    def test_avg_aliases_above_three(self, skills):
        m1 = check_m1(skills)
        avg = m1["avg_aliases_per_skill"]
        assert avg >= 3.0, f"Average aliases per skill {avg} < 3.0 — corpus under-specified"

    def test_no_skill_has_zero_aliases(self, skills):
        empty = [s.canonical for s in skills if not s.aliases]
        assert not empty, f"Skills with no aliases: {empty}"

    def test_docker_ce_alias_exists(self, skills):
        docker = next((s for s in skills if s.canonical == "docker"), None)
        assert docker is not None, "docker skill not found in CSV"
        all_aliases_lower = [a.lower() for a in docker.aliases]
        assert "docker ce" in all_aliases_lower, (
            "'docker ce' alias missing from docker entry in skills_master.csv"
        )

    def test_nodejs_canonical_exists(self, skills):
        canonical_set = {s.canonical for s in skills}
        assert "nodejs" in canonical_set, (
            "'nodejs' canonical missing from skills_master.csv — "
            "Node.js is referenced in role profiles and ground truth fixtures"
        )


# ─────────────────── TestAliasCollisions ─────────────────────────────────────

class TestAliasCollisions:
    """
    Alias collisions: two canonicals claiming the same alias string.
    Many are benign dual-labels (e.g. 'istio' is both a skill and a service_mesh alias).
    The test hard-fails only on unambiguously problematic collisions in _CRITICAL_COLLISIONS.
    All collisions are still visible in the validate_corpus.py report (M5).
    """

    # Aliases that are genuinely ambiguous and cause wrong extractions
    _CRITICAL_COLLISIONS = {"tf"}  # terraform vs tensorflow — context-dependent

    def test_no_critical_alias_collisions(self, skills):
        _, collisions = build_alias_map(skills)
        collision_map = {c["alias"]: c["claimed_by"] for c in collisions}
        critical_found = {
            alias: claimants
            for alias, claimants in collision_map.items()
            if alias in self._CRITICAL_COLLISIONS
        }
        if critical_found:
            details = "\n".join(
                f"  '{alias}' claimed by: {claimants}"
                for alias, claimants in critical_found.items()
            )
            pytest.fail(
                f"Critical alias collision(s) found — last-writer wins silently:\n{details}\n"
                f"Fix: remove the ambiguous alias from one skill's CSV entry."
            )

    def test_total_collision_count_reported(self, skills):
        """Informational: print collision count. Not a hard failure."""
        _, collisions = build_alias_map(skills)
        if collisions:
            print(
                f"\n[INFO] M5: {len(collisions)} alias collision(s) detected "
                f"(run validate_corpus.py --verbose for full list)"
            )


# ─────────────────── TestSeedIntegrity ───────────────────────────────────────

class TestSeedIntegrity:
    """All skill canonicals in seed_db.py must exist in skills_master.csv."""

    def test_no_broken_skill_refs(self, skills):
        m6 = check_m6(skills)
        if m6["broken_count"] > 0:
            broken = "\n".join(f"  - {ref}" for ref in m6["broken_refs"])
            pytest.fail(
                f"seed_db.py ROLE_SKILL_PROFILES references {m6['broken_count']} "
                f"canonical(s) not found in skills_master.csv:\n{broken}\n\n"
                f"Either add the canonical to skills_master.csv or correct the "
                f"reference in ROLE_SKILL_PROFILES."
            )
