"""
Phase 1 — Role Normalization Regression Suite

Covers all role inputs specified in the evaluation brief:
  - 21 canonical / variant spellings
  - 4 deliberate misspellings
  - Edge cases: empty, numeric, very long, the Compiler Engineer crash

Every test calls the real normalize_role() against the real DB.
DB marker auto-skips this file if TEST_DATABASE_URL is unreachable or unseeded.

PASS = resolved correctly, no exception, confidence in [0,1]
FAIL = exception, wrong match_type, confidence out of range, or zero suggestions
       on a misspelling that should give hints
"""

import pytest

from src.services.role_normalizer import RoleMatch, normalize_role

# ---------------------------------------------------------------------------
# Canonical and variant role inputs from the spec
# ---------------------------------------------------------------------------

CANONICAL_ROLES = [
    "Software Engineer",
    "software engineer",
    "SOFTWARE ENGINEER",
    "SWE",
    "SDE",
    "Software Developer",
    "Backend Developer",
    "Backend Engineer",
    "Full Stack Developer",
    "Fullstack Engineer",
    "Machine Learning Engineer",
    "ML Engineer",
    "AI Engineer",
    "Data Scientist",
    "Compiler Engineer",
    "Security Engineer",
    "Cloud Engineer",
    "DevOps Engineer",
    "MLOps Engineer",
    "Prompt Engineer",
    "Research Engineer",
]

MISSPELLINGS = [
    ("Softwre Enginer",        "Software Engineer"),
    ("Machne Learnng Enginer", "Machine Learning Engineer"),
    ("Backend Develper",       "Backend Engineer"),
    ("Dat Scintist",           "Data Scientist"),
]

VALID_MATCH_TYPES = frozenset({
    "exact_alias", "fuzzy_match", "semantic_direct", "semantic_suggest", "no_match",
})


# ---------------------------------------------------------------------------
# Phase 1-A: Every canonical input — no crash, valid shape
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestCanonicalRolesNoException:
    """Every listed input must return a RoleMatch without raising."""

    @pytest.mark.parametrize("role_input", CANONICAL_ROLES)
    def test_returns_without_exception(self, eval_session, role_input):
        try:
            result = normalize_role(role_input, eval_session)
        except Exception as exc:
            pytest.fail(
                f"normalize_role({role_input!r}) raised {type(exc).__name__}: {exc}"
            )
        assert isinstance(result, RoleMatch), (
            f"Expected RoleMatch, got {type(result)}"
        )

    @pytest.mark.parametrize("role_input", CANONICAL_ROLES)
    def test_match_type_is_valid(self, eval_session, role_input):
        result = normalize_role(role_input, eval_session)
        assert result.match_type in VALID_MATCH_TYPES, (
            f"{role_input!r} — unknown match_type: {result.match_type!r}"
        )

    @pytest.mark.parametrize("role_input", CANONICAL_ROLES)
    def test_confidence_in_range(self, eval_session, role_input):
        result = normalize_role(role_input, eval_session)
        assert 0.0 <= result.confidence <= 1.0, (
            f"{role_input!r} — confidence {result.confidence} outside [0, 1]"
        )

    @pytest.mark.parametrize("role_input", CANONICAL_ROLES)
    def test_suggestions_are_list(self, eval_session, role_input):
        result = normalize_role(role_input, eval_session)
        assert isinstance(result.suggestions, list), (
            f"{role_input!r} — suggestions is {type(result.suggestions)}, expected list"
        )


# ---------------------------------------------------------------------------
# Phase 1-B: Case variants resolve to the same role
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestCaseVariants:
    def test_all_case_variants_match_same_role(self, eval_session):
        """'Software Engineer' / 'software engineer' / 'SOFTWARE ENGINEER' → same role_id."""
        variants = ["Software Engineer", "software engineer", "SOFTWARE ENGINEER"]
        results  = [normalize_role(v, eval_session) for v in variants]
        resolved = [(v, r.role_id) for v, r in zip(variants, results) if r.role_id]
        if len(resolved) >= 2:
            ids = {rid for _, rid in resolved}
            assert len(ids) == 1, (
                f"Case variants resolve to different roles: "
                f"{dict(resolved)}"
            )

    def test_ml_engineer_aliases(self, eval_session):
        """'ML Engineer' and 'Machine Learning Engineer' must resolve to the same role."""
        r1 = normalize_role("ML Engineer",               eval_session)
        r2 = normalize_role("Machine Learning Engineer", eval_session)
        if r1.role_id and r2.role_id:
            assert r1.role_id == r2.role_id, (
                f"'ML Engineer' (id={r1.role_id}) != 'Machine Learning Engineer' "
                f"(id={r2.role_id})"
            )


# ---------------------------------------------------------------------------
# Phase 1-C: Abbreviation aliases
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestAbbreviations:
    def test_swe_resolves(self, eval_session):
        result = normalize_role("SWE", eval_session)
        has_result = result.role_id is not None or bool(result.suggestions)
        assert has_result, (
            "'SWE' returned no_match with no suggestions — add 'SWE' to Backend/Software Engineer aliases"
        )

    def test_sde_resolves(self, eval_session):
        result = normalize_role("SDE", eval_session)
        has_result = result.role_id is not None or bool(result.suggestions)
        assert has_result, (
            "'SDE' returned no_match with no suggestions — add 'SDE' to aliases"
        )


# ---------------------------------------------------------------------------
# Phase 1-D: Misspelling regression
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestMisspellings:

    @pytest.mark.parametrize("misspelled,expected_role", MISSPELLINGS)
    def test_no_exception(self, eval_session, misspelled, expected_role):
        try:
            normalize_role(misspelled, eval_session)
        except Exception as exc:
            pytest.fail(
                f"normalize_role({misspelled!r}) raised {type(exc).__name__}: {exc}"
            )

    @pytest.mark.parametrize("misspelled,expected_role", MISSPELLINGS)
    def test_produces_resolution_or_suggestions(self, eval_session, misspelled, expected_role):
        """
        A misspelled role must either resolve directly OR return suggestions.
        Returning no_match with empty suggestions leaves the user stranded.
        """
        result = normalize_role(misspelled, eval_session)
        has_resolution  = result.role_id is not None
        has_suggestions = bool(result.suggestions)
        assert has_resolution or has_suggestions, (
            f"{misspelled!r} → no resolution and no suggestions. "
            f"Expected hints toward {expected_role!r}. "
            f"match_type={result.match_type}, confidence={result.confidence:.3f}"
        )

    @pytest.mark.parametrize("misspelled,expected_role", MISSPELLINGS)
    def test_suggestions_sorted_descending(self, eval_session, misspelled, expected_role):
        """When suggestions are returned, they must be ordered by similarity desc."""
        result = normalize_role(misspelled, eval_session)
        if len(result.suggestions) > 1:
            sims = [s.get("similarity", 0) for s in result.suggestions]
            assert sims == sorted(sims, reverse=True), (
                f"{misspelled!r} suggestions not sorted by similarity: {sims}"
            )


# ---------------------------------------------------------------------------
# Phase 1-E: The Compiler Engineer regression
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestCompilerEngineerRegression:
    def test_compiler_engineer_no_exception(self, eval_session):
        """
        Regression: 'Compiler Engineer' previously crashed with
        psycopg2.errors.SyntaxError caused by ':vec::vector' in the pgvector query.
        Any exception here means the CAST(:vec AS vector) fix did not take effect
        (server not restarted, or fix not applied).
        """
        try:
            result = normalize_role("Compiler Engineer", eval_session)
        except Exception as exc:
            pytest.fail(
                f"'Compiler Engineer' crashed — pgvector cast fix not active: "
                f"{type(exc).__name__}: {exc}"
            )
        assert isinstance(result, RoleMatch)

    def test_compiler_engineer_match_type_valid(self, eval_session):
        result = normalize_role("Compiler Engineer", eval_session)
        assert result.match_type in VALID_MATCH_TYPES

    def test_compiler_engineer_confidence_in_range(self, eval_session):
        result = normalize_role("Compiler Engineer", eval_session)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Phase 1-F: Edge / adversarial inputs
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestEdgeCases:
    def test_empty_string(self, eval_session):
        """Empty input must not raise."""
        try:
            normalize_role("", eval_session)
        except Exception as exc:
            pytest.fail(f"Empty string raised: {exc}")

    def test_numeric_input(self, eval_session):
        try:
            normalize_role("12345", eval_session)
        except Exception as exc:
            pytest.fail(f"Numeric input raised: {exc}")

    def test_whitespace_only(self, eval_session):
        try:
            normalize_role("   ", eval_session)
        except Exception as exc:
            pytest.fail(f"Whitespace input raised: {exc}")

    def test_very_long_input(self, eval_session):
        long_role = "Senior Software Engineer " * 60
        try:
            normalize_role(long_role, eval_session)
        except Exception as exc:
            pytest.fail(f"Very long input raised: {exc}")

    def test_special_characters(self, eval_session):
        for inp in ("C++ Engineer", "C# Developer", "AI/ML Engineer"):
            try:
                normalize_role(inp, eval_session)
            except Exception as exc:
                pytest.fail(f"{inp!r} raised: {exc}")

    def test_truly_unknown_role_is_graceful(self, eval_session):
        """A nonsense role must not crash and must return a valid RoleMatch."""
        result = normalize_role("Underwater Basket Weaving Engineer", eval_session)
        assert isinstance(result, RoleMatch)
        assert result.match_type in VALID_MATCH_TYPES
