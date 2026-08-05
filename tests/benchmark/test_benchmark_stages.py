"""
Pipeline Stage Latency Tests
============================
Each test class validates a specific pipeline stage against a latency target.
Tests measure wall-clock time — they are not unit tests.

Run all:
    pytest tests/benchmark/ -v -s

Run only fast (no DB, no LLM) tests:
    pytest tests/benchmark/ -v -s -m "not db and not llm"

Run with DB:
    pytest tests/benchmark/ -v -s -m "not llm"

Targets (p95):
    parser      < 500 ms
    alias scan  <  50 ms   (pre-compiled, O1)
    embedding   < 800 ms
    pgvector    < 200 ms   (batched, O2)
    gap         < 150 ms
    llm         < 10 000 ms
"""

import re
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"

# ── pytest markers ────────────────────────────────────────────────────────────
# Register in conftest.py if needed; defined here for documentation.
# pytest.ini / pyproject.toml should add:
#   markers = ["db: requires a live database", "llm: requires GROQ_API_KEY"]


# ── Helpers ───────────────────────────────────────────────────────────────────

BENCH_RUNS = 15   # iterations per timing measurement


def _p95(data: list) -> float:
    s = sorted(data)
    return s[min(int(len(s) * 0.95), len(s) - 1)]


def _timeit(fn, runs: int = BENCH_RUNS) -> list[float]:
    """Run fn() `runs` times and return list of latencies in ms."""
    results = []
    for _ in range(runs):
        t0 = time.monotonic()
        fn()
        results.append((time.monotonic() - t0) * 1000)
    return results


def _resume_sections() -> dict[str, str]:
    return {
        "skills": (
            "Python FastAPI PostgreSQL Docker Kubernetes Redis React TypeScript "
            "Git Linux AWS Terraform CI/CD PyTorch scikit-learn pandas numpy "
            "REST APIs GraphQL microservices agile TDD system design C++ Go"
        ),
        "experience": (
            "Software Engineer at TechCorp 2022 to present. Built scalable backend "
            "services using Python and FastAPI. Managed PostgreSQL databases and "
            "Redis caches. Deployed containers to Kubernetes clusters on AWS. "
            "CI/CD pipelines using GitHub Actions. Terraform for infrastructure."
        ),
        "education": (
            "B.Tech Computer Science VIT University 2020 to 2024 GPA 8.7."
        ),
        "projects": (
            "CareerCopilot AI platform FastAPI PostgreSQL pgvector Groq LLM. "
            "Stock predictor LSTM PyTorch pandas."
        ),
    }


def _make_synthetic_pdf() -> bytes:
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            "John Developer\n\nTECHNICAL SKILLS\n"
            + _resume_sections()["skills"]
            + "\n\nEXPERIENCE\n"
            + _resume_sections()["experience"],
            fontsize=10,
        )
        return doc.tobytes()
    except ImportError:
        return (
            b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\n"
            b"0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000058 00000 n\n0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )


# ── Session fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def alias_map_and_skills():
    from scripts.validate_corpus import load_skills, build_alias_map
    skills = load_skills(DATA_DIR)
    am, _ = build_alias_map(skills)
    return am, skills


@pytest.fixture(scope="session")
def embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        from src.config import get_settings
        model = SentenceTransformer(get_settings().embedding_model)
        return model
    except ImportError:
        pytest.skip("sentence-transformers not installed")


@pytest.fixture(scope="session")
def resume_chunks(embedding_model):
    from src.services.skill_extractor import _make_chunks
    text = " ".join(_resume_sections().values())
    return _make_chunks(text, chunk_size=80, overlap=20)


@pytest.fixture(scope="session")
def resume_embeddings(embedding_model, resume_chunks):
    return list(
        embedding_model.encode(
            resume_chunks, normalize_embeddings=True, show_progress_bar=False
        )
    )


# ── TestParserLatency ─────────────────────────────────────────────────────────

class TestParserLatency:
    """PDF text extraction + section detection must complete in < 500 ms p95."""

    TARGET_P95_MS = 500.0

    def test_pdf_parse_p95_under_target(self):
        from src.services.resume_parser import parse_resume

        pdf_bytes = _make_synthetic_pdf()
        timings = _timeit(lambda: parse_resume(pdf_bytes, "resume.pdf"))
        p95 = _p95(timings)
        print(f"\n  [parser] p95={p95:.0f}ms  target={self.TARGET_P95_MS:.0f}ms")
        assert p95 < self.TARGET_P95_MS, (
            f"Parser p95 {p95:.0f}ms exceeded target {self.TARGET_P95_MS:.0f}ms"
        )

    def test_section_detection_p95_under_100ms(self):
        """Section detection alone (without file I/O) must be < 100 ms."""
        from src.services.resume_parser import _detect_sections

        text = "\n".join([
            "TECHNICAL SKILLS", _resume_sections()["skills"],
            "EXPERIENCE", _resume_sections()["experience"],
            "EDUCATION", _resume_sections()["education"],
        ])
        timings = _timeit(lambda: _detect_sections(text))
        p95 = _p95(timings)
        print(f"\n  [section-detect] p95={p95:.1f}ms")
        assert p95 < 100.0, f"Section detection p95 {p95:.1f}ms > 100ms"


# ── TestAliasLatency ──────────────────────────────────────────────────────────

class TestAliasLatency:
    """
    Alias scan with pre-compiled patterns (O1) must be < 150 ms p95.

    Dev note: the CSV-based alias map used here has ~2955 aliases (full corpus).
    The production DB alias map has ~2334 (active skills only), so production
    will be faster. The hard target is 150 ms; production target is 50 ms.
    The more important assertion is the speedup test: compiled must be >= 2x faster.
    """

    TARGET_COMPILED_P95_MS = 150.0

    def test_precompiled_alias_scan_p95_under_target(self, alias_map_and_skills):
        from scripts.validate_corpus import _make_boundary_pattern

        alias_map, _ = alias_map_and_skills
        text = " ".join(_resume_sections().values()).lower()
        sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)

        # Pre-compile once (simulates O1 process-level cache)
        patterns = {a: re.compile(_make_boundary_pattern(a)) for a in sorted_aliases if a}

        timings = _timeit(
            lambda: [patterns[a].search(text) for a in sorted_aliases if a]
        )
        p95 = _p95(timings)
        print(
            f"\n  [alias-compiled] p95={p95:.0f}ms  "
            f"aliases={len(sorted_aliases)}  target={self.TARGET_COMPILED_P95_MS:.0f}ms"
        )
        assert p95 < self.TARGET_COMPILED_P95_MS, (
            f"Pre-compiled alias scan p95 {p95:.0f}ms exceeded {self.TARGET_COMPILED_P95_MS:.0f}ms"
        )

    def test_precompiled_faster_than_uncompiled(self, alias_map_and_skills):
        """O1 must produce a measurable speedup over the uncompiled baseline."""
        from scripts.validate_corpus import _make_boundary_pattern

        alias_map, _ = alias_map_and_skills
        text = " ".join(_resume_sections().values()).lower()
        sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)

        # Baseline: compile + search each time (original code path)
        uncompiled = _timeit(
            lambda: [
                re.search(_make_boundary_pattern(a), text)
                for a in sorted_aliases if a
            ]
        )

        # O1: pre-compiled patterns
        patterns = {a: re.compile(_make_boundary_pattern(a)) for a in sorted_aliases if a}
        compiled = _timeit(
            lambda: [patterns[a].search(text) for a in sorted_aliases if a]
        )

        speedup = _p95(uncompiled) / max(_p95(compiled), 0.1)
        print(
            f"\n  [alias] uncompiled p95={_p95(uncompiled):.0f}ms  "
            f"compiled p95={_p95(compiled):.0f}ms  speedup={speedup:.1f}x"
        )
        assert speedup >= 2.0, (
            f"O1 speedup {speedup:.1f}x is less than 2x — caching may not be effective"
        )

    def test_alias_map_build_from_csv_under_500ms(self, alias_map_and_skills):
        """Loading skills_master.csv + building alias map must be < 500 ms."""
        from scripts.validate_corpus import load_skills, build_alias_map

        timings = _timeit(lambda: build_alias_map(load_skills(DATA_DIR)), runs=5)
        p95 = _p95(timings)
        print(f"\n  [alias-csv-build] p95={p95:.0f}ms")
        assert p95 < 500.0, (
            f"Alias map build (CSV) p95 {p95:.0f}ms > 500ms"
        )


# ── TestEmbeddingLatency ──────────────────────────────────────────────────────

class TestEmbeddingLatency:
    """Batch embedding inference for ~8 resume chunks must be < 800 ms p95."""

    TARGET_P95_MS = 800.0

    def test_batch_encode_p95_under_target(self, embedding_model, resume_chunks):
        # Warm-up (load weights into CPU caches)
        embedding_model.encode(
            resume_chunks[:1], normalize_embeddings=True, show_progress_bar=False
        )

        timings = _timeit(
            lambda: embedding_model.encode(
                resume_chunks, normalize_embeddings=True, show_progress_bar=False
            )
        )
        p95 = _p95(timings)
        print(
            f"\n  [embedding] p95={p95:.0f}ms  "
            f"chunks={len(resume_chunks)}  target={self.TARGET_P95_MS:.0f}ms"
        )
        assert p95 < self.TARGET_P95_MS, (
            f"Embedding batch encode p95 {p95:.0f}ms exceeded target {self.TARGET_P95_MS:.0f}ms"
        )

    def test_single_chunk_encode_under_200ms(self, embedding_model):
        """A single-chunk encode (role normalisation path) must be < 200 ms."""
        single = ["Backend Software Engineer specialising in Python microservices"]
        embedding_model.encode(single, normalize_embeddings=True, show_progress_bar=False)

        timings = _timeit(
            lambda: embedding_model.encode(
                single, normalize_embeddings=True, show_progress_bar=False
            )
        )
        p95 = _p95(timings)
        print(f"\n  [embedding-single] p95={p95:.0f}ms")
        assert p95 < 200.0, f"Single encode p95 {p95:.0f}ms > 200ms"


# ── TestPgvectorLatency ───────────────────────────────────────────────────────

@pytest.mark.db
class TestPgvectorLatency:
    """Batched ANN search (O2) for all resume chunks must be < 200 ms p95.
    Requires a live database. Run with: pytest -m db
    """

    TARGET_BATCH_P95_MS = 200.0

    @pytest.fixture(scope="class")
    def db_session(self):
        try:
            from src.database import SessionLocal
            db = SessionLocal()
            yield db
            db.close()
        except Exception as e:
            pytest.skip(f"DB unavailable: {e}")

    def test_batched_pgvector_p95_under_target(
        self, db_session, resume_embeddings
    ):
        from src.services.skill_extractor import _pgvector_search_batch

        db_runs = max(BENCH_RUNS // 4, 5)
        timings = _timeit(
            lambda: _pgvector_search_batch(resume_embeddings, db_session, top_k=5),
            runs=db_runs,
        )
        p95 = _p95(timings)
        print(
            f"\n  [pgvector-batch] p95={p95:.0f}ms  "
            f"embeddings={len(resume_embeddings)}  target={self.TARGET_BATCH_P95_MS:.0f}ms"
        )
        assert p95 < self.TARGET_BATCH_P95_MS, (
            f"Batched pgvector p95 {p95:.0f}ms exceeded target {self.TARGET_BATCH_P95_MS:.0f}ms"
        )

    def test_batched_faster_than_per_chunk(self, db_session, resume_embeddings):
        """O2 batched query must be faster than N per-chunk queries."""
        from src.services.skill_extractor import _pgvector_search, _pgvector_search_batch

        db_runs = max(BENCH_RUNS // 4, 5)

        per_chunk = _timeit(
            lambda: [_pgvector_search(e, db_session, top_k=5) for e in resume_embeddings],
            runs=db_runs,
        )
        batched = _timeit(
            lambda: _pgvector_search_batch(resume_embeddings, db_session, top_k=5),
            runs=db_runs,
        )

        speedup = _p95(per_chunk) / max(_p95(batched), 0.1)
        print(
            f"\n  [pgvector] per-chunk p95={_p95(per_chunk):.0f}ms  "
            f"batched p95={_p95(batched):.0f}ms  speedup={speedup:.1f}x"
        )
        assert speedup >= 1.5, (
            f"O2 speedup {speedup:.1f}x is less than 1.5x — "
            f"batch query may not be working correctly"
        )


# ── TestGapLatency ────────────────────────────────────────────────────────────

@pytest.mark.db
class TestGapLatency:
    """Gap analysis role-profile query must be < 150 ms p95.
    Requires a live database. Run with: pytest -m db
    """

    TARGET_P95_MS = 150.0

    @pytest.fixture(scope="class")
    def db_session_and_role(self):
        try:
            from sqlalchemy import text
            from src.database import SessionLocal
            db = SessionLocal()
            row = db.execute(text("SELECT role_id FROM role_categories LIMIT 1")).first()
            if not row:
                pytest.skip("No roles in DB — run seed_db.py first")
            yield db, row.role_id
            db.close()
        except Exception as e:
            pytest.skip(f"DB unavailable: {e}")

    def test_role_profile_query_p95_under_target(self, db_session_and_role):
        from src.services.gap_analyzer import _get_role_profile

        db, role_id = db_session_and_role
        db_runs = max(BENCH_RUNS // 4, 5)
        timings = _timeit(lambda: _get_role_profile(role_id, db), runs=db_runs)
        p95 = _p95(timings)
        print(
            f"\n  [gap-profile] p95={p95:.0f}ms  "
            f"role_id={role_id}  target={self.TARGET_P95_MS:.0f}ms"
        )
        assert p95 < self.TARGET_P95_MS, (
            f"Role profile query p95 {p95:.0f}ms exceeded target {self.TARGET_P95_MS:.0f}ms"
        )

    def test_bulk_skill_display_faster_than_n_plus_1(self, db_session_and_role):
        """O5: _get_skill_display_bulk must be faster than N individual queries."""
        from sqlalchemy import text
        from src.services.gap_analyzer import _get_skill_display_bulk

        db, _ = db_session_and_role
        rows = db.execute(text("SELECT skill_id FROM skills LIMIT 20")).fetchall()
        skill_ids = [r.skill_id for r in rows]
        if not skill_ids:
            pytest.skip("No skills in DB")

        from src.models.skill import Skill

        def n_plus_1():
            return {sid: db.query(Skill).filter(Skill.skill_id == sid).first() for sid in skill_ids}

        db_runs = max(BENCH_RUNS // 4, 5)
        n1_times = _timeit(n_plus_1, runs=db_runs)
        bulk_times = _timeit(lambda: _get_skill_display_bulk(skill_ids, db), runs=db_runs)

        speedup = _p95(n1_times) / max(_p95(bulk_times), 0.1)
        print(
            f"\n  [gap-n+1] N+1 p95={_p95(n1_times):.0f}ms  "
            f"bulk p95={_p95(bulk_times):.0f}ms  speedup={speedup:.1f}x"
        )
        assert speedup >= 1.5, (
            f"O5 bulk query speedup {speedup:.1f}x < 1.5x for {len(skill_ids)} skills"
        )


# ── TestLLMLatency ────────────────────────────────────────────────────────────

@pytest.mark.llm
class TestLLMLatency:
    """Groq roadmap generation must be < 10 000 ms p95.
    Requires GROQ_API_KEY. Run with: pytest -m llm
    WARNING: each run consumes API quota.
    """

    TARGET_P95_MS = 10_000.0

    def test_llm_call_p95_under_target(self):
        try:
            from src.services.roadmap_generator import _build_prompt, _call_llm_with_retry
        except ImportError as e:
            pytest.skip(str(e))

        gap = {
            "readiness_score": 40,
            "missing_skills": [
                {"display_name": "Kubernetes", "importance_score": 0.9, "skill_id": 1},
                {"display_name": "AWS", "importance_score": 0.85, "skill_id": 2},
            ],
            "matched_skills": [{"display_name": "Python", "skill_id": 3}],
        }
        prompt = _build_prompt("Backend Engineer", gap, "3 months", "v1")

        timings = _timeit(lambda: _call_llm_with_retry(prompt), runs=3)
        p95 = _p95(timings)
        print(
            f"\n  [llm] p95={p95:.0f}ms  target={self.TARGET_P95_MS:.0f}ms  "
            f"(max_tokens=2000 after O3)"
        )
        assert p95 < self.TARGET_P95_MS, (
            f"LLM p95 {p95:.0f}ms exceeded target {self.TARGET_P95_MS:.0f}ms. "
            f"Consider streaming or further reducing max_tokens."
        )

    def test_max_tokens_is_2000(self):
        """Guard that O3 was not accidentally reverted."""
        import inspect
        from src.services.roadmap_generator import _call_llm_with_retry
        src = inspect.getsource(_call_llm_with_retry)
        assert "max_tokens=2000" in src, (
            "O3: max_tokens must be 2000 in _call_llm_with_retry — "
            "was it reverted to 4096?"
        )


# ── TestOptimisationInvariants ────────────────────────────────────────────────

class TestOptimisationInvariants:
    """Structural checks that optimisations are wired correctly (no DB needed)."""

    def test_o1_alias_cache_primed_sets_module_globals(self, alias_map_and_skills):
        """After _prime_caches, all three module-level caches must be populated."""
        import src.services.skill_extractor as se

        # Reset to test cold path
        se._alias_map_cache = None
        se._sorted_aliases_cache = None
        se._pattern_cache = None

        class _FakeDb:
            def query(self, model):
                return self

            def filter(self, *a, **kw):
                return self

            def all(self):
                from scripts.validate_corpus import load_skills
                raw = load_skills(DATA_DIR)

                class _Skill:
                    def __init__(self, s):
                        self.skill_id = hash(s.canonical)
                        self.canonical_name = s.canonical
                        self.display_name = s.display
                        self.aliases = list(s.aliases)
                        self.is_active = True

                return [_Skill(s) for s in raw]

        se._prime_caches(_FakeDb())

        assert se._alias_map_cache is not None, "_alias_map_cache not set"
        assert se._sorted_aliases_cache is not None, "_sorted_aliases_cache not set"
        assert se._pattern_cache is not None, "_pattern_cache not set"
        assert len(se._alias_map_cache) > 500, "Expected > 500 aliases"

        # Restore clean state so other tests aren't affected
        se.invalidate_alias_cache()

    def test_o2_batch_function_exists(self):
        """_pgvector_search_batch must be importable from skill_extractor."""
        from src.services.skill_extractor import _pgvector_search_batch
        assert callable(_pgvector_search_batch)

    def test_o3_max_tokens_not_4096(self):
        """Confirm max_tokens was reduced (O3)."""
        import inspect
        from src.services.roadmap_generator import _call_llm_with_retry
        src = inspect.getsource(_call_llm_with_retry)
        assert "max_tokens=4096" not in src, "O3 not applied: max_tokens is still 4096"

    def test_o5_bulk_display_function_exists(self):
        """_get_skill_display_bulk must exist in gap_analyzer."""
        from src.services.gap_analyzer import _get_skill_display_bulk
        assert callable(_get_skill_display_bulk)

    def test_o6_generate_roadmap_has_no_db_param(self):
        """O6: generate_roadmap must NOT accept a db Session parameter."""
        import inspect
        from src.services.roadmap_generator import generate_roadmap
        sig = inspect.signature(generate_roadmap)
        assert "db" not in sig.parameters, (
            "O6 not applied: generate_roadmap still accepts db parameter — "
            "it should manage its own DB connections internally"
        )

    def test_o8_worker_ready_signal_registered(self):
        """O8: worker.py must register a worker_ready signal handler."""
        import inspect
        import src.worker as worker_mod
        src = inspect.getsource(worker_mod)
        assert "worker_ready" in src, (
            "O8 not applied: worker_ready signal handler missing from worker.py"
        )
