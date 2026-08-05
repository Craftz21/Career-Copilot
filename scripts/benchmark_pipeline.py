#!/usr/bin/env python3
"""
CareerCopilot — Pipeline Performance Benchmark
===============================================
Measures per-stage latency and compares against the 15-second E2E target.

Stages measured (most require no DB or Groq key):
  parser     — PDF/DOCX text extraction + section detection
  alias      — alias map build (cold) vs cached + regex scan
  embedding  — sentence-transformers batch inference over resume chunks
  pgvector   — ANN vector search, per-chunk vs batched  (needs --with-db)
  gap        — gap analysis role-profile query           (needs --with-db)
  llm        — Groq roadmap generation                  (needs --with-llm)

Usage:
  python scripts/benchmark_pipeline.py
  python scripts/benchmark_pipeline.py --stage alias --runs 50
  python scripts/benchmark_pipeline.py --with-db --with-llm
  python scripts/benchmark_pipeline.py --format json --output reports/bench.json
"""

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
FIXTURE_DIR = ROOT / "test_resumes" / "ground_truth"

# ── Per-stage latency targets (p95, milliseconds) ────────────────────────────
TARGETS_MS: dict[str, float] = {
    "parser":    500.0,
    "alias":     150.0,   # full CSV has ~2955 aliases; production DB has ~2334 (target 50ms)
    "embedding": 800.0,
    "pgvector":  200.0,   # for the full batch of ~8 chunks
    "gap":       150.0,
    "llm":    10_000.0,
}
# Warn when p95 exceeds 80 % of target (approaching limit)
_WARN_RATIO = 0.80


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class StageResult:
    name: str
    target_ms: float
    runs: list = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    notes: list = field(default_factory=list)  # extra measurements / findings

    @property
    def p50(self) -> float:
        return statistics.median(self.runs) if self.runs else 0.0

    @property
    def p95(self) -> float:
        if not self.runs:
            return 0.0
        s = sorted(self.runs)
        return s[min(int(len(s) * 0.95), len(s) - 1)]

    @property
    def p99(self) -> float:
        if not self.runs:
            return 0.0
        s = sorted(self.runs)
        return s[min(int(len(s) * 0.99), len(s) - 1)]

    @property
    def mean(self) -> float:
        return statistics.mean(self.runs) if self.runs else 0.0

    @property
    def status(self) -> str:
        if self.skipped or not self.runs:
            return "SKIP"
        if self.p95 > self.target_ms:
            return "FAIL"
        if self.p95 > self.target_ms * _WARN_RATIO:
            return "WARN"
        return "PASS"


# ── Benchmark runner ──────────────────────────────────────────────────────────

class PipelineBenchmark:
    def __init__(self, runs: int = 20, with_db: bool = False, with_llm: bool = False):
        self.runs = runs
        self.with_db = with_db
        self.with_llm = with_llm
        self._db = None

    # ── Shared inputs ─────────────────────────────────────────────────────────

    def _get_file_bytes(self) -> tuple[bytes, str]:
        """Return (bytes, filename) — tries real fixture first, synthesises PDF fallback."""
        for gt in FIXTURE_DIR.glob("*.json"):
            for ext in (".pdf", ".docx"):
                candidate = ROOT / "test_resumes" / (gt.stem + ext)
                if candidate.exists():
                    return candidate.read_bytes(), candidate.name
        return _make_synthetic_pdf(), "benchmark_resume.pdf"

    def _get_test_sections(self) -> dict[str, str]:
        return {
            "skills": (
                "Python FastAPI PostgreSQL Docker Kubernetes Redis React TypeScript "
                "Git Linux AWS Terraform CI/CD PyTorch scikit-learn pandas numpy "
                "REST APIs GraphQL microservices agile TDD system design C++ Go"
            ),
            "experience": (
                "Software Engineer at TechCorp 2022 to present. Built scalable backend "
                "services using Python and FastAPI. Managed PostgreSQL databases and Redis "
                "caches. Deployed containers to Kubernetes clusters on AWS. Implemented "
                "CI/CD pipelines using GitHub Actions. Led migration from monolith to "
                "microservices architecture. Contributed to open-source projects. Used "
                "Terraform for infrastructure as code. Worked with React TypeScript frontend."
            ),
            "education": (
                "B.Tech Computer Science VIT University 2020 to 2024 GPA 8.7 out of 10. "
                "Courses Data Structures Algorithms Operating Systems DBMS Computer Networks."
            ),
            "projects": (
                "CareerCopilot AI powered career intelligence platform FastAPI PostgreSQL "
                "pgvector Groq LLM deployed on Render with Neon and Upstash. "
                "Stock predictor LSTM model using PyTorch and pandas on NSE data."
            ),
        }

    def _get_db(self):
        if self._db is None:
            from src.database import SessionLocal
            self._db = SessionLocal()
        return self._db

    # ── Stage: parser ─────────────────────────────────────────────────────────

    def bench_parser(self) -> StageResult:
        from src.services.resume_parser import parse_resume

        file_bytes, filename = self._get_file_bytes()
        result = StageResult(name="parser", target_ms=TARGETS_MS["parser"])

        for _ in range(self.runs):
            t0 = time.monotonic()
            try:
                parse_resume(file_bytes, filename)
            except Exception:
                pass
            result.runs.append((time.monotonic() - t0) * 1000)

        return result

    # ── Stage: alias ──────────────────────────────────────────────────────────

    def bench_alias(self) -> StageResult:
        from scripts.validate_corpus import _make_boundary_pattern, load_skills, build_alias_map

        skills = load_skills(DATA_DIR)
        alias_map, _ = build_alias_map(skills)
        text = " ".join(self._get_test_sections().values()).lower()
        sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
        alias_count = len(sorted_aliases)

        result = StageResult(name="alias", target_ms=TARGETS_MS["alias"])

        # ── A: uncompiled (current behaviour before O1) ───────────────────────
        uncompiled = []
        for _ in range(self.runs):
            t0 = time.monotonic()
            for alias in sorted_aliases:
                if alias:
                    re.search(_make_boundary_pattern(alias), text)
            uncompiled.append((time.monotonic() - t0) * 1000)

        # ── B: pre-compiled (O1 behaviour) ────────────────────────────────────
        patterns = {a: re.compile(_make_boundary_pattern(a)) for a in sorted_aliases if a}
        compiled = []
        for _ in range(self.runs):
            t0 = time.monotonic()
            for alias in sorted_aliases:
                if alias:
                    patterns[alias].search(text)
            compiled.append((time.monotonic() - t0) * 1000)

        result.runs = compiled  # target is measured against O1 (compiled)
        result.notes = [
            f"Aliases scanned: {alias_count}",
            f"Uncompiled p95: {_p95(uncompiled):.0f} ms  (before O1)",
            f"Pre-compiled p95: {_p95(compiled):.0f} ms  (after O1)",
            f"Speedup: {_p95(uncompiled) / max(_p95(compiled), 0.1):.1f}×",
        ]
        return result

    # ── Stage: embedding ──────────────────────────────────────────────────────

    def bench_embedding(self) -> StageResult:
        try:
            from sentence_transformers import SentenceTransformer
            from src.config import get_settings
            from src.services.skill_extractor import _make_chunks
        except ImportError as e:
            return StageResult(
                name="embedding", target_ms=TARGETS_MS["embedding"],
                skipped=True, skip_reason=str(e),
            )

        settings = get_settings()
        model = SentenceTransformer(settings.embedding_model)
        text = " ".join(self._get_test_sections().values())
        chunks = _make_chunks(text, chunk_size=80, overlap=20)

        # Warm-up pass (fills CPU caches, avoids cold-start in first timing)
        model.encode(chunks[:1], normalize_embeddings=True, show_progress_bar=False)

        result = StageResult(name="embedding", target_ms=TARGETS_MS["embedding"])
        for _ in range(self.runs):
            t0 = time.monotonic()
            model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
            result.runs.append((time.monotonic() - t0) * 1000)

        result.notes = [
            f"Chunks encoded per run: {len(chunks)}",
            f"Embedding model: {settings.embedding_model}",
        ]
        return result

    # ── Stage: pgvector ───────────────────────────────────────────────────────

    def bench_pgvector(self) -> StageResult:
        if not self.with_db:
            return StageResult(
                name="pgvector", target_ms=TARGETS_MS["pgvector"],
                skipped=True, skip_reason="pass --with-db to enable (requires live DB)",
            )
        try:
            from sentence_transformers import SentenceTransformer
            from src.config import get_settings
            from src.services.skill_extractor import (
                _make_chunks,
                _pgvector_search,
                _pgvector_search_batch,
            )
            import numpy as np

            db = self._get_db()
            settings = get_settings()
            model = SentenceTransformer(settings.embedding_model)
            text = " ".join(self._get_test_sections().values())
            chunks = _make_chunks(text, chunk_size=80, overlap=20)
            embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
            emb_list = list(embeddings)
            db_runs = max(self.runs // 4, 5)

            # ── A: per-chunk (N queries — before O2) ──────────────────────────
            per_chunk = []
            for _ in range(db_runs):
                t0 = time.monotonic()
                for emb in emb_list:
                    _pgvector_search(emb, db, top_k=5)
                per_chunk.append((time.monotonic() - t0) * 1000)

            # ── B: batched (1 query — O2) ─────────────────────────────────────
            batched = []
            for _ in range(db_runs):
                t0 = time.monotonic()
                _pgvector_search_batch(emb_list, db, top_k=5)
                batched.append((time.monotonic() - t0) * 1000)

            result = StageResult(name="pgvector", target_ms=TARGETS_MS["pgvector"])
            result.runs = batched  # target measured against O2 (batched)
            result.notes = [
                f"Chunk count: {len(chunks)}",
                f"Per-chunk p95: {_p95(per_chunk):.0f} ms  ({len(chunks)} queries — before O2)",
                f"Batched p95: {_p95(batched):.0f} ms  (1 query — after O2)",
                f"Speedup: {_p95(per_chunk) / max(_p95(batched), 0.1):.1f}×",
            ]
            return result

        except Exception as e:
            return StageResult(
                name="pgvector", target_ms=TARGETS_MS["pgvector"],
                skipped=True, skip_reason=f"DB error: {e}",
            )

    # ── Stage: gap analysis ───────────────────────────────────────────────────

    def bench_gap(self) -> StageResult:
        if not self.with_db:
            return StageResult(
                name="gap", target_ms=TARGETS_MS["gap"],
                skipped=True, skip_reason="pass --with-db to enable (requires live DB)",
            )
        try:
            from sqlalchemy import text as sql_text
            from src.services.gap_analyzer import _get_role_profile

            db = self._get_db()
            row = db.execute(sql_text("SELECT role_id FROM role_categories LIMIT 1")).first()
            if not row:
                return StageResult(
                    name="gap", target_ms=TARGETS_MS["gap"],
                    skipped=True, skip_reason="No roles in DB — run seed_db.py first",
                )
            role_id = row.role_id
            db_runs = max(self.runs // 4, 5)

            result = StageResult(name="gap", target_ms=TARGETS_MS["gap"])
            for _ in range(db_runs):
                t0 = time.monotonic()
                _get_role_profile(role_id, db)
                result.runs.append((time.monotonic() - t0) * 1000)

            result.notes = [f"Role ID benchmarked: {role_id}"]
            return result

        except Exception as e:
            return StageResult(
                name="gap", target_ms=TARGETS_MS["gap"],
                skipped=True, skip_reason=f"DB error: {e}",
            )

    # ── Stage: LLM ───────────────────────────────────────────────────────────

    def bench_llm(self) -> StageResult:
        if not self.with_llm:
            return StageResult(
                name="llm", target_ms=TARGETS_MS["llm"],
                skipped=True,
                skip_reason="pass --with-llm to enable (consumes Groq API quota)",
            )
        try:
            from src.services.roadmap_generator import _build_prompt, _call_llm_with_retry

            sample_gap = {
                "readiness_score": 45,
                "missing_skills": [
                    {"display_name": "Kubernetes", "importance_score": 0.9, "skill_id": 1},
                    {"display_name": "AWS", "importance_score": 0.85, "skill_id": 2},
                    {"display_name": "Terraform", "importance_score": 0.75, "skill_id": 3},
                    {"display_name": "Go", "importance_score": 0.70, "skill_id": 4},
                ],
                "matched_skills": [
                    {"display_name": "Python", "skill_id": 5},
                    {"display_name": "Docker", "skill_id": 6},
                ],
            }
            prompt = _build_prompt("Backend Software Engineer", sample_gap, "3 months", "v1")

            llm_runs = min(self.runs, 3)  # LLM is expensive — cap at 3 calls
            result = StageResult(name="llm", target_ms=TARGETS_MS["llm"])
            for _ in range(llm_runs):
                t0 = time.monotonic()
                content, model_used = _call_llm_with_retry(prompt)
                result.runs.append((time.monotonic() - t0) * 1000)

            result.notes = [
                f"Model: {model_used}",
                f"max_tokens: 2000 (O3 — was 4096)",
                f"Runs: {llm_runs} (capped to limit API quota)",
            ]
            return result

        except Exception as e:
            return StageResult(
                name="llm", target_ms=TARGETS_MS["llm"],
                skipped=True, skip_reason=f"LLM error: {e}",
            )

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run(self, stages: Optional[list[str]] = None) -> list[StageResult]:
        runners = {
            "parser":    self.bench_parser,
            "alias":     self.bench_alias,
            "embedding": self.bench_embedding,
            "pgvector":  self.bench_pgvector,
            "gap":       self.bench_gap,
            "llm":       self.bench_llm,
        }
        active = {k: v for k, v in runners.items() if not stages or k in stages}

        results = []
        for name, fn in active.items():
            print(f"  [{name:10s}] running...", end=" ", flush=True)
            r = fn()
            sym = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "SKIP"}
            if r.skipped:
                print(f"SKIP  ({r.skip_reason})")
            else:
                print(f"{sym[r.status]}  p95={r.p95:.0f}ms  target={r.target_ms:.0f}ms")
            results.append(r)

        if self._db:
            self._db.close()
        return results


# ── Report renderers ──────────────────────────────────────────────────────────

def render_markdown(results: list[StageResult]) -> str:
    lines = [
        "# CareerCopilot — Pipeline Performance Benchmark",
        "",
        "## Stage Latency",
        "",
        "| Stage | Target p95 (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Runs | Status |",
        "|-------|-----------------|----------|----------|----------|------|--------|",
    ]

    for r in results:
        if r.skipped:
            lines.append(
                f"| {r.name} | {r.target_ms:.0f} | — | — | — | — | SKIP |"
            )
        else:
            lines.append(
                f"| {r.name} | {r.target_ms:.0f} | "
                f"{r.p50:.0f} | {r.p95:.0f} | {r.p99:.0f} | "
                f"{len(r.runs)} | {r.status} |"
            )

    # E2E estimate (sum of non-LLM non-skipped stages)
    excl_llm = [r for r in results if not r.skipped and r.name != "llm"]
    if excl_llm:
        e2e_p95 = sum(r.p95 for r in excl_llm)
        e2e_status = "PASS" if e2e_p95 < 5_000 else "FAIL"
        lines.append(
            f"| **e2e excl. LLM** | 5000 | — | {e2e_p95:.0f} | — | — | {e2e_status} |"
        )
        llm_result = next((r for r in results if r.name == "llm" and not r.skipped), None)
        if llm_result:
            total = e2e_p95 + llm_result.p95
            total_status = "PASS" if total < 15_000 else "FAIL"
            lines.append(
                f"| **e2e total** | 15000 | — | {total:.0f} | — | — | {total_status} |"
            )

    # Notes per stage
    notes_lines = ["", "## Stage Notes", ""]
    for r in results:
        if r.notes:
            notes_lines.append(f"### {r.name}")
            for note in r.notes:
                notes_lines.append(f"- {note}")
            notes_lines.append("")

    # Optimisation summary
    opt_lines = [
        "## Optimisations Applied",
        "",
        "| Code | File | Change | Expected Impact |",
        "|------|------|--------|-----------------|",
        "| O1 | skill_extractor.py | Alias map + 2334 regex patterns cached at worker startup | alias scan −80% |",
        "| O2 | skill_extractor.py | pgvector: N per-chunk queries → 1 batched query (json_to_recordset+LATERAL) | pgvector −85% |",
        "| O3 | roadmap_generator.py | max_tokens 4096 → 2000 | LLM −30%–50% |",
        "| O4 | analyze_resume.py | _update_task calls 7 → 4 (save 3 DB connections per run) | −3 conn/run |",
        "| O5 | gap_analyzer.py | N+1 bonus-skill display queries → 1 bulk query | gap −variable |",
        "| O6 | analyze_resume.py + roadmap_generator.py | DB transaction split; LLM runs outside open context | −5–15 s conn hold |",
        "| O7 | (SQL — not yet applied) | CREATE INDEX on roadmaps((content->>_cache_key)) | cache lookup −95% |",
        "| O8 | worker.py | Embedding model pre-warmed at worker_ready signal | cold-start −2–10 s |",
        "",
        "## Pending Recommendations",
        "",
        "- **O7**: `CREATE INDEX idx_roadmaps_cache_key ON roadmaps ((content->>'_cache_key'))` — add to next migration.",
        "- **Streaming**: For cache-miss paths, stream LLM tokens to the client (SSE) to cut perceived latency.",
        "- **Async worker**: Migrate to `celery[gevent]` or `asyncio` to allow concurrent I/O within a single worker slot.",
        "",
    ]

    return "\n".join(lines + notes_lines + opt_lines)


def render_json(results: list[StageResult]) -> str:
    return json.dumps(
        {
            "stages": [
                {
                    "name": r.name,
                    "target_ms": r.target_ms,
                    "p50_ms": round(r.p50, 1),
                    "p95_ms": round(r.p95, 1),
                    "p99_ms": round(r.p99, 1),
                    "mean_ms": round(r.mean, 1),
                    "runs": len(r.runs),
                    "status": r.status,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                    "notes": r.notes,
                }
                for r in results
            ]
        },
        indent=2,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _p95(data: list) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * 0.95), len(s) - 1)]


def _make_synthetic_pdf() -> bytes:
    """Synthesise a minimal PDF with realistic resume text using PyMuPDF."""
    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            """John Developer | john@example.com

TECHNICAL SKILLS
Python, FastAPI, PostgreSQL, Docker, Kubernetes, Redis, React, TypeScript,
Git, Linux, AWS, Terraform, CI/CD, PyTorch, scikit-learn, pandas, numpy,
REST APIs, GraphQL, microservices, agile, TDD, system design, C++, Go

EXPERIENCE
Software Engineer — TechCorp (2022–present)
Built scalable backend services using Python and FastAPI.
Managed PostgreSQL databases with pgvector for semantic search.
Deployed containers to Kubernetes clusters on AWS.
CI/CD pipelines with GitHub Actions. Terraform for IaC.

EDUCATION
B.Tech Computer Science, VIT University (2020–2024), GPA 8.7

PROJECTS
CareerCopilot: AI career platform (FastAPI, PostgreSQL, pgvector, Groq LLM)
Stock Predictor: LSTM model using PyTorch on NSE data""",
            fontsize=10,
        )
        return doc.tobytes()
    except ImportError:
        # Fallback: bare-minimum valid PDF
        return (
            b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\n"
            b"0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000058 00000 n\n0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CareerCopilot pipeline benchmark")
    parser.add_argument(
        "--stage",
        choices=["parser", "alias", "embedding", "pgvector", "gap", "llm"],
        help="Run only this stage (default: all)",
    )
    parser.add_argument("--runs", type=int, default=20, help="Timing iterations (default 20)")
    parser.add_argument("--with-db", action="store_true", help="Enable DB-dependent stages")
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM stage (uses API quota)")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", help="Write report to this file path")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else None

    print(f"\nCareerCopilot Benchmark  ({args.runs} runs per stage)\n")
    bench = PipelineBenchmark(runs=args.runs, with_db=args.with_db, with_llm=args.with_llm)
    results = bench.run(stages=stages)

    if args.format == "json":
        report = render_json(results)
    else:
        report = render_markdown(results)

    # Windows console safe — replace non-ASCII
    safe = report.encode("ascii", errors="replace").decode("ascii")

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.output}")
    else:
        print("\n" + safe)

    # Exit 1 if any non-skipped stage failed
    failed = [r for r in results if r.status == "FAIL"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
