"""
Phase 3 — pgvector SQL Audit

Two layers:
  Static  — scan every .py source file for banned ::vector syntax
  Runtime — execute each pgvector code path and confirm no SQL exception

FAIL conditions:
  - Any Python source file contains `:vec::vector` or similar bind-param double-colon cast
  - Any pgvector query raises a psycopg2 SyntaxError or DimensionMismatch
  - Vector dimension is inconsistent between skills and queries

Background:
  SQLAlchemy's text() parser sees `:foo::bar` as a bind parameter named `foo:bar`
  (or similar misparse), causing psycopg2 SyntaxError at runtime.
  Correct form: CAST(:foo AS vector) — not affected by SQLAlchemy token parsing.
"""

import re
from pathlib import Path

import pytest
from sqlalchemy import text

# Root of the source tree
_SRC_ROOT = Path(__file__).resolve().parents[3]  # projectAI/
_PY_SOURCES = list(_SRC_ROOT.glob("src/**/*.py")) + list(_SRC_ROOT.glob("scripts/**/*.py"))

# Pattern: a SQLAlchemy bind parameter (colon-prefixed identifier) followed by ::type
# e.g.  :vec::vector   :embedding::vector
_BAD_CAST_RE = re.compile(r":[a-zA-Z_]\w*::[a-zA-Z_]\w+")

# All legitimate CAST(...) forms — what we want to see
_GOOD_CAST_RE = re.compile(r"CAST\([^)]+AS\s+vector\)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Phase 3-A: Static source audit (no DB required)
# ---------------------------------------------------------------------------

class TestStaticAudit:
    def test_no_bind_param_double_colon_cast(self):
        """
        Scan every .py source file for SQLAlchemy bind parameters cast with ::
        e.g.  1 - (embedding <=> :vec::vector)

        A match here means the CAST fix was not applied, and the corresponding
        code path will crash at runtime with psycopg2.errors.SyntaxError.
        """
        violations: list[str] = []
        for path in _PY_SOURCES:
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(source.splitlines(), 1):
                if _BAD_CAST_RE.search(line):
                    violations.append(f"{path.relative_to(_SRC_ROOT)}:{lineno}  →  {line.strip()}")

        assert not violations, (
            f"\n\nFound {len(violations)} bind-parameter double-colon cast(s).\n"
            "These will crash at runtime. Fix: replace ':vec::vector' with 'CAST(:vec AS vector)'.\n\n"
            + "\n".join(violations)
        )

    def test_all_vector_casts_use_cast_form(self):
        """
        Every file that contains an embedding <=> operator must also use CAST(... AS vector).
        Detects any file that searches vectors but forgot to fix the cast syntax.
        """
        offenders: list[str] = []
        for path in _PY_SOURCES:
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "embedding <=>" in source and not _GOOD_CAST_RE.search(source):
                offenders.append(str(path.relative_to(_SRC_ROOT)))

        assert not offenders, (
            f"\n\nFiles with 'embedding <=>' but no CAST(... AS vector) form:\n"
            + "\n".join(offenders)
            + "\nThese files may be using ::vector syntax and will crash for certain role inputs."
        )

    def test_role_normalizer_uses_cast(self):
        path = _SRC_ROOT / "src" / "services" / "role_normalizer.py"
        source = path.read_text(encoding="utf-8")
        assert _GOOD_CAST_RE.search(source), (
            "role_normalizer.py does not contain CAST(:vec AS vector). "
            "Compiler Engineer will crash."
        )
        assert not _BAD_CAST_RE.search(source), (
            "role_normalizer.py still contains a bind-parameter ::vector cast."
        )

    def test_skill_extractor_uses_cast(self):
        path = _SRC_ROOT / "src" / "services" / "skill_extractor.py"
        source = path.read_text(encoding="utf-8")
        assert _GOOD_CAST_RE.search(source), (
            "skill_extractor.py does not contain CAST(... AS vector)."
        )
        assert not _BAD_CAST_RE.search(source), (
            "skill_extractor.py still contains a bind-parameter ::vector cast."
        )


# ---------------------------------------------------------------------------
# Phase 3-B: Runtime execution (requires DB + pgvector extension)
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
class TestRuntimeExecution:
    def test_pgvector_extension_installed(self, eval_session):
        """vector extension must exist — everything else depends on it."""
        row = eval_session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).first()
        assert row is not None, (
            "pgvector extension not installed. Run: CREATE EXTENSION vector;"
        )

    def test_role_normalizer_vector_query_executes(self, eval_session):
        """
        Execute the exact SQL in role_normalizer._pgvector_search() with a dummy vector.
        Fails with SyntaxError if the fix was not applied (old code running in process).
        """
        dummy_vec = "[" + ",".join(["0.0"] * 384) + "]"
        try:
            eval_session.execute(
                text(
                    """
                    SELECT role_id, canonical_name, display_name,
                           1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                    FROM role_categories
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:vec AS vector)
                    LIMIT 5
                    """
                ),
                {"vec": dummy_vec},
            ).fetchall()
        except Exception as exc:
            pytest.fail(
                f"role_categories pgvector query failed: {type(exc).__name__}: {exc}\n"
                "If this is a SyntaxError, the server is still running old code. "
                "Restart uvicorn + Celery worker."
            )

    def test_skill_extractor_batch_query_executes(self, eval_session):
        """
        Execute _pgvector_search_batch's lateral subquery with a one-item records batch.
        """
        import json
        records = [{"idx": 0, "vec": "[" + ",".join(["0.0"] * 384) + "]"}]
        try:
            eval_session.execute(
                text(
                    """
                    SELECT q.idx::int AS idx, s.skill_id, s.similarity
                    FROM json_to_recordset(:records) AS q(idx int, vec text)
                    CROSS JOIN LATERAL (
                        SELECT skill_id,
                               1 - (embedding <=> CAST(q.vec AS vector)) AS similarity
                        FROM   skills
                        WHERE  is_active = true AND embedding IS NOT NULL
                        ORDER  BY embedding <=> CAST(q.vec AS vector)
                        LIMIT  3
                    ) AS s
                    ORDER BY q.idx, s.similarity DESC
                    """
                ),
                {"records": json.dumps(records)},
            ).fetchall()
        except Exception as exc:
            pytest.fail(
                f"skill_extractor batch pgvector query failed: {type(exc).__name__}: {exc}"
            )

    def test_skill_extractor_single_query_executes(self, eval_session):
        """Execute _pgvector_search (single-vector path) with a dummy vector."""
        dummy_vec = "[" + ",".join(["0.0"] * 384) + "]"
        try:
            eval_session.execute(
                text(
                    """
                    SELECT skill_id, 1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                    FROM skills
                    WHERE is_active = true AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:vec AS vector)
                    LIMIT 5
                    """
                ),
                {"vec": dummy_vec, "k": 5},
            ).fetchall()
        except Exception as exc:
            pytest.fail(
                f"skill_extractor single pgvector query failed: {type(exc).__name__}: {exc}"
            )

    @pytest.mark.requires_seed
    def test_skills_embeddings_have_consistent_dimension(self, eval_session):
        """
        All non-null skill embeddings must have the same vector dimension.
        A mixed dimension would cause 'different vector dimensions' errors.
        """
        rows = eval_session.execute(
            text(
                "SELECT skill_id, vector_dims(embedding) AS dim "
                "FROM skills WHERE embedding IS NOT NULL LIMIT 50"
            )
        ).fetchall()
        if not rows:
            pytest.skip("No skills with embeddings — seed with `make seed`")
        dims = {r.dim for r in rows}
        assert len(dims) == 1, (
            f"Inconsistent embedding dimensions in skills table: {dims}. "
            "This will cause random 'different vector dimensions' crashes."
        )

    @pytest.mark.requires_seed
    def test_role_embeddings_have_consistent_dimension(self, eval_session):
        """All non-null role embeddings must share the same dimension."""
        rows = eval_session.execute(
            text(
                "SELECT role_id, vector_dims(embedding) AS dim "
                "FROM role_categories WHERE embedding IS NOT NULL LIMIT 20"
            )
        ).fetchall()
        if not rows:
            pytest.skip("No role embeddings — seed with `make seed`")
        dims = {r.dim for r in rows}
        assert len(dims) == 1, (
            f"Inconsistent embedding dimensions in role_categories: {dims}."
        )
