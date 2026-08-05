#!/usr/bin/env python3
"""
Resume Parser Evaluation Script
================================
Measures parser quality against a labelled corpus of resume fixtures.

Usage:
    python scripts/evaluate_parser.py \\
        --resume-dir test_resumes/ \\
        --ground-truth-dir test_resumes/ground_truth/ \\
        [--class single_column] \\
        [--format markdown|json] \\
        [--output eval_report.md] \\
        [--verbose]

No database or running server required.
Skill matching uses data/skills_master.csv for alias lookup (text-mode).
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.services.resume_parser import ParseError, parse_resume


# ─────────────────── Data classes ────────────────────────────────────────────

@dataclass
class GroundTruth:
    resume_id: str
    resume_class: str
    layout_type: str
    difficulty: str
    min_char_count: int
    max_char_count: int
    expected_sections: list
    must_not_contain_sections: list
    expected_skills: list       # [{display_name, canonical, in_section}]
    expected_education: list    # [{institution_contains, degree_contains, year}]
    expected_experience: list   # [{company_contains, title_contains}]
    known_issues: Optional[str]
    evaluator_notes: Optional[str]


@dataclass
class ResumeRecord:
    path: Path
    gt: GroundTruth


@dataclass
class ParseAttempt:
    record: ResumeRecord
    success: bool
    result: Optional[dict]
    error_type: Optional[str]
    error_msg: Optional[str]
    latency_ms: float


@dataclass
class MetricSet:
    resume_id: str
    resume_class: str
    difficulty: str
    # Parse outcome
    parse_success: bool
    latency_ms: float
    error_type: Optional[str]
    error_msg: Optional[str]
    # Section detection
    sdr: Optional[float]
    sfdr: Optional[float]
    detected_sections: list = field(default_factory=list)
    # Skill metrics
    skill_precision: Optional[float] = None
    skill_recall: Optional[float] = None
    skill_f1: Optional[float] = None
    matched_skills: list = field(default_factory=list)
    missed_skills: list = field(default_factory=list)
    false_positive_skills: list = field(default_factory=list)
    # Text quality
    tqs: Optional[float] = None
    tqs_details: dict = field(default_factory=dict)
    # Education / experience
    edu_accuracy: Optional[float] = None
    exp_accuracy: Optional[float] = None
    # Metadata
    known_issues: Optional[str] = None


# ─────────────────── Ground truth loading ────────────────────────────────────

def load_ground_truth(gt_path: Path) -> GroundTruth:
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)

    ep = data.get("expected_parse", {})
    return GroundTruth(
        resume_id=data["resume_id"],
        resume_class=data["resume_class"],
        layout_type=data["layout_type"],
        difficulty=data.get("difficulty", "medium"),
        min_char_count=ep.get("min_char_count", 100),
        max_char_count=ep.get("max_char_count", 50000),
        expected_sections=ep.get("expected_sections", []),
        must_not_contain_sections=ep.get("must_not_contain_sections", []),
        expected_skills=data.get("expected_skills", []),
        expected_education=data.get("expected_education", []),
        expected_experience=data.get("expected_experience", []),
        known_issues=data.get("known_issues"),
        evaluator_notes=data.get("evaluator_notes"),
    )


def discover_corpus(
    resume_dir: Path,
    gt_dir: Path,
    filter_class: Optional[str] = None,
) -> list[ResumeRecord]:
    classes = (
        [filter_class]
        if filter_class
        else ["single_column", "double_column", "ats", "canva", "docx", "image_heavy"]
    )
    records: list[ResumeRecord] = []

    for cls in classes:
        cls_resume_dir = resume_dir / cls
        cls_gt_dir = gt_dir / cls
        if not cls_resume_dir.exists():
            continue

        for resume_path in sorted(cls_resume_dir.iterdir()):
            if resume_path.suffix.lower() not in (".pdf", ".docx", ".doc"):
                continue
            gt_path = cls_gt_dir / (resume_path.stem + ".json")
            if not gt_path.exists():
                print(
                    f"[WARN] No ground truth for {cls}/{resume_path.name} — skipping",
                    file=sys.stderr,
                )
                continue
            records.append(ResumeRecord(path=resume_path, gt=load_ground_truth(gt_path)))

    return records


# ─────────────────── Alias map (CSV, no DB) ──────────────────────────────────

def load_alias_map(data_dir: Path) -> dict[str, tuple[str, str]]:
    """
    Return {alias_lower: (canonical_name, display_name)} from skills_master.csv.
    Enables text-mode skill precision/recall without a live database.
    """
    skills_csv = data_dir / "skills_master.csv"
    if not skills_csv.exists():
        print(
            f"[WARN] {skills_csv} not found — skill metrics will be unavailable",
            file=sys.stderr,
        )
        return {}

    alias_map: dict[str, tuple[str, str]] = {}
    with open(skills_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            canonical = row["canonical_name"].strip()
            display = row["display_name"].strip()
            alias_map[canonical.lower()] = (canonical, display)
            alias_map[display.lower()] = (canonical, display)
            for alias in (row.get("aliases") or "").split("|"):
                alias = alias.strip().lower()
                if alias:
                    alias_map[alias] = (canonical, display)

    return alias_map


def _make_boundary_pattern(alias: str) -> str:
    r"""
    \b fails when the alias starts or ends with a non-word char (c++, c#, f#).
    Use negative lookarounds in those cases.
    """
    escaped = re.escape(alias)
    prefix = r"\b" if alias[0].isalnum() or alias[0] == "_" else r"(?<!\w)"
    suffix = r"\b" if alias[-1].isalnum() or alias[-1] == "_" else r"(?!\w)"
    return prefix + escaped + suffix


def scan_text_for_skills(text: str, alias_map: dict) -> set[str]:
    """
    Return canonical skill names found in text using word-boundary matching.
    Uses lookarounds for aliases that start/end with non-word chars (c++, c#, f#).
    """
    if not text or not alias_map:
        return set()

    text_lower = text.lower()
    found: set[str] = set()

    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if not alias:
            continue
        if re.search(_make_boundary_pattern(alias), text_lower):
            canonical, _ = alias_map[alias]
            found.add(canonical)

    return found


# ─────────────────── Metric computations ─────────────────────────────────────

def compute_section_metrics(
    detected: list[str],
    expected: list[str],
    must_not: list[str],
) -> tuple[float, float]:
    det = set(detected)
    exp = set(expected)

    sdr = len(det & exp) / len(exp) if exp else 1.0
    false_sects = (det - exp) | (det & set(must_not))
    sfdr = len(false_sects) / len(det) if det else 0.0

    return round(sdr, 4), round(sfdr, 4)


def compute_skill_metrics(
    raw_text: str,
    gt: GroundTruth,
    alias_map: dict,
) -> dict:
    if not alias_map or not gt.expected_skills:
        return {
            "precision": None, "recall": None, "f1": None,
            "matched": [], "missed": [], "false_positives": [],
        }

    extracted = scan_text_for_skills(raw_text, alias_map)
    expected = {s["canonical"] for s in gt.expected_skills}
    tp = extracted & expected

    precision = len(tp) / len(extracted) if extracted else 0.0
    recall = len(tp) / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": sorted(tp),
        "missed": sorted(expected - extracted),
        "false_positives": sorted(extracted - expected),
    }


def compute_tqs(raw_text: str, gt: GroundTruth) -> tuple[float, dict]:
    """Text Quality Score: composite of 5 binary sub-checks."""
    checks: dict[str, bool] = {}

    checks["min_char_count"] = len(raw_text) >= gt.min_char_count
    checks["max_char_count"] = len(raw_text) <= gt.max_char_count

    if raw_text:
        printable = sum(1 for c in raw_text if c.isprintable() or c in "\n\r\t")
        checks["coherent"] = (printable / len(raw_text)) >= 0.80
    else:
        checks["coherent"] = False

    checks["no_excessive_whitespace"] = not re.search(r"\n{5,}", raw_text)

    word_count = len(re.findall(r"[a-zA-Z]{3,}", raw_text))
    checks["has_meaningful_words"] = word_count >= 10

    score = round(sum(checks.values()) / len(checks), 4)
    return score, checks


def compute_edu_accuracy(sections: dict, gt: GroundTruth) -> Optional[float]:
    if not gt.expected_education:
        return None
    search_text = (sections.get("education") or "").lower()
    if not search_text:
        return None
    matched = 0
    for edu in gt.expected_education:
        ok = all(
            not edu.get(k) or edu[k].lower() in search_text
            for k in ("institution_contains", "degree_contains")
        )
        if ok:
            matched += 1
    return round(matched / len(gt.expected_education), 4)


def compute_exp_accuracy(sections: dict, gt: GroundTruth) -> Optional[float]:
    if not gt.expected_experience:
        return None
    search_text = (sections.get("experience") or "").lower()
    if not search_text:
        return None
    matched = 0
    for exp in gt.expected_experience:
        ok = all(
            not exp.get(k) or exp[k].lower() in search_text
            for k in ("company_contains", "title_contains")
        )
        if ok:
            matched += 1
    return round(matched / len(gt.expected_experience), 4)


# ─────────────────── Parse & evaluate ────────────────────────────────────────

def run_parse(record: ResumeRecord) -> ParseAttempt:
    t0 = time.perf_counter()
    try:
        result = parse_resume(record.path.read_bytes(), record.path.name)
        return ParseAttempt(
            record=record, success=True, result=result,
            error_type=None, error_msg=None,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
    except Exception as exc:
        return ParseAttempt(
            record=record, success=False, result=None,
            error_type=type(exc).__name__, error_msg=str(exc),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )


def evaluate_one(attempt: ParseAttempt, alias_map: dict) -> MetricSet:
    gt = attempt.record.gt

    if not attempt.success:
        return MetricSet(
            resume_id=gt.resume_id,
            resume_class=gt.resume_class,
            difficulty=gt.difficulty,
            parse_success=False,
            latency_ms=attempt.latency_ms,
            error_type=attempt.error_type,
            error_msg=attempt.error_msg,
            sdr=None,
            sfdr=None,
            known_issues=gt.known_issues,
        )

    result = attempt.result
    raw_text = result["raw_text"]
    sections = result["sections"]
    detected_sections = list(sections.keys())

    sdr, sfdr = compute_section_metrics(
        detected_sections, gt.expected_sections, gt.must_not_contain_sections
    )
    skill_m = compute_skill_metrics(raw_text, gt, alias_map)
    tqs, tqs_details = compute_tqs(raw_text, gt)

    return MetricSet(
        resume_id=gt.resume_id,
        resume_class=gt.resume_class,
        difficulty=gt.difficulty,
        parse_success=True,
        latency_ms=attempt.latency_ms,
        error_type=None,
        error_msg=None,
        sdr=sdr,
        sfdr=sfdr,
        detected_sections=detected_sections,
        skill_precision=skill_m["precision"],
        skill_recall=skill_m["recall"],
        skill_f1=skill_m["f1"],
        matched_skills=skill_m["matched"],
        missed_skills=skill_m["missed"],
        false_positive_skills=skill_m["false_positives"],
        tqs=tqs,
        tqs_details=tqs_details,
        edu_accuracy=compute_edu_accuracy(sections, gt),
        exp_accuracy=compute_exp_accuracy(sections, gt),
        known_issues=gt.known_issues,
    )


# ─────────────────── Aggregation ─────────────────────────────────────────────

def _mean(values: list) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _pct(values: list, p: int) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return round(s[idx], 1)


def aggregate(metrics: list[MetricSet]) -> dict:
    by_class: dict[str, list[MetricSet]] = defaultdict(list)
    for m in metrics:
        by_class[m.resume_class].append(m)

    class_reports = {}
    for cls, cls_m in by_class.items():
        succ = [m for m in cls_m if m.parse_success]
        total = len(cls_m)
        failed = total - len(succ)
        class_reports[cls] = {
            "total": total,
            "failed": failed,
            "failure_rate": round(failed / total, 4) if total else 0.0,
            "sdr_avg": _mean([m.sdr for m in succ]),
            "sfdr_avg": _mean([m.sfdr for m in succ]),
            "skill_precision_avg": _mean([m.skill_precision for m in succ]),
            "skill_recall_avg": _mean([m.skill_recall for m in succ]),
            "skill_f1_avg": _mean([m.skill_f1 for m in succ]),
            "tqs_avg": _mean([m.tqs for m in succ]),
            "edu_accuracy_avg": _mean([m.edu_accuracy for m in succ]),
            "exp_accuracy_avg": _mean([m.exp_accuracy for m in succ]),
            "latency_p50_ms": _pct([m.latency_ms for m in cls_m], 50),
            "latency_p95_ms": _pct([m.latency_ms for m in cls_m], 95),
        }

    all_succ = [m for m in metrics if m.parse_success]
    return {
        "overall": {
            "total": len(metrics),
            "failed": sum(1 for m in metrics if not m.parse_success),
            "failure_rate": round(
                sum(1 for m in metrics if not m.parse_success) / len(metrics), 4
            ) if metrics else 0.0,
            "sdr_avg": _mean([m.sdr for m in all_succ]),
            "skill_f1_avg": _mean([m.skill_f1 for m in all_succ]),
            "tqs_avg": _mean([m.tqs for m in all_succ]),
        },
        "by_class": class_reports,
    }


# ─────────────────── Diagnostics ─────────────────────────────────────────────

def find_common_mistakes(metrics: list[MetricSet]) -> list[str]:
    mistakes = []

    # False section headers
    bleeding = [m for m in metrics if m.parse_success and m.sfdr is not None and m.sfdr > 0.3]
    if bleeding:
        classes = sorted({m.resume_class for m in bleeding})
        mistakes.append(
            f"[Section bleed] SFDR > 0.3 in {len(bleeding)} resume(s) across: "
            f"{', '.join(classes)} — short bullet lines promoted to section headers "
            f"(resume_parser.py:149 len<40 check too broad)"
        )

    # Consistently missed skills
    low_recall = [m for m in metrics if m.parse_success and m.skill_recall is not None and m.skill_recall < 0.5]
    if low_recall:
        all_missed = [s for m in low_recall for s in m.missed_skills]
        top = Counter(all_missed).most_common(5)
        if top:
            missed_str = ", ".join(f"{s}({n}x)" for s, n in top)
            mistakes.append(
                f"[Low recall] Skills consistently missed: {missed_str} — "
                f"check that canonical names in ground truth match skills_master.csv"
            )

    # Frequent false positive skills
    high_fp = [m for m in metrics if m.parse_success and m.false_positive_skills]
    if high_fp:
        all_fp = [s for m in high_fp for s in m.false_positive_skills]
        top = Counter(all_fp).most_common(5)
        if top:
            fp_str = ", ".join(f"{s}({n}x)" for s, n in top)
            mistakes.append(
                f"[False positives] Frequently over-extracted: {fp_str} — "
                f"alias substring too broad or threshold too low"
            )

    # Unexpected ParseError (not image_heavy)
    unexpected = [m for m in metrics if not m.parse_success and m.resume_class != "image_heavy"]
    if unexpected:
        ids = ", ".join(m.resume_id for m in unexpected)
        mistakes.append(f"[Unexpected ParseError] Non-image resumes failed: {ids}")

    # Low section detection
    low_sdr = [m for m in metrics if m.parse_success and m.sdr is not None and m.sdr < 0.5]
    if low_sdr:
        classes = sorted({m.resume_class for m in low_sdr})
        mistakes.append(
            f"[Low SDR] Section detection < 0.5 in {len(low_sdr)} resume(s) "
            f"({', '.join(classes)}) — non-standard headers not matching regex"
        )

    return mistakes


def worst_n(metrics: list[MetricSet], n: int = 5) -> list[MetricSet]:
    failures = [m for m in metrics if not m.parse_success]
    successes = sorted(
        [m for m in metrics if m.parse_success],
        key=lambda m: m.skill_f1 if m.skill_f1 is not None else 0.0,
    )
    return (failures + successes)[:n]


# ─────────────────── Report rendering ────────────────────────────────────────

def _fmt(v: Optional[float], decimals: int = 3) -> str:
    return f"{v:.{decimals}f}" if v is not None else "  -  "


def render_markdown(metrics: list[MetricSet], agg: dict, verbose: bool = False) -> str:
    o = agg["overall"]
    lines: list[str] = []

    lines += [
        f"# Resume Parser Evaluation Report — {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"**Corpus:** {o['total']} resumes | "
        f"**Failures:** {o['failed']} ({o['failure_rate']*100:.1f}%)",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Section Detection Rate (macro-avg) | {_fmt(o['sdr_avg'])} |",
        f"| Skill F1 (macro-avg)               | {_fmt(o['skill_f1_avg'])} |",
        f"| Text Quality Score (macro-avg)      | {_fmt(o['tqs_avg'])} |",
        f"| Parse Failure Rate                 | {o['failure_rate']*100:.1f}% |",
        "",
        "## Per-Class Breakdown",
        "",
        "| Class          | N | Fail% |  SDR  | Prec  |  Rec  |  F1   |  TQS  | P95 ms |",
        "|----------------|---|-------|-------|-------|-------|-------|-------|--------|",
    ]

    for cls, r in sorted(agg["by_class"].items()):
        p95 = str(r["latency_p95_ms"]) if r["latency_p95_ms"] is not None else "  -"
        lines.append(
            f"| {cls:<14} | {r['total']} | {r['failure_rate']*100:>5.1f}% "
            f"| {_fmt(r['sdr_avg'])} | {_fmt(r['skill_precision_avg'])} "
            f"| {_fmt(r['skill_recall_avg'])} | {_fmt(r['skill_f1_avg'])} "
            f"| {_fmt(r['tqs_avg'])} | {p95:>6} |"
        )

    lines += ["", "## Failures", ""]
    failures = [m for m in metrics if not m.parse_success]
    if failures:
        for m in failures:
            tag = " _(expected)_" if m.resume_class == "image_heavy" else ""
            lines.append(
                f"- **{m.resume_id}** ({m.resume_class}) → "
                f"`{m.error_type}`: {m.error_msg}{tag}"
            )
    else:
        lines.append("_No failures._")

    lines += ["", "## Common Parser Mistakes", ""]
    mistakes = find_common_mistakes(metrics)
    if mistakes:
        for i, m in enumerate(mistakes, 1):
            lines.append(f"{i}. {m}")
    else:
        lines.append("_No common mistakes detected._")

    lines += ["", "## Worst Resumes by Skill F1", ""]
    worst = worst_n(metrics)
    if worst:
        lines += [
            "| Resume | Class | Difficulty | F1 | SDR | Issue |",
            "|--------|-------|------------|----|-----|-------|",
        ]
        for m in worst:
            f1 = _fmt(m.skill_f1) if m.skill_f1 is not None else "FAIL"
            sdr = _fmt(m.sdr) if m.sdr is not None else "FAIL"
            issue = m.error_type if not m.parse_success else (m.known_issues or "")
            lines.append(
                f"| {m.resume_id} | {m.resume_class} | {m.difficulty} "
                f"| {f1} | {sdr} | {issue} |"
            )

    if verbose:
        lines += ["", "## Per-Resume Detail", ""]
        for m in sorted(metrics, key=lambda x: (x.resume_class, x.resume_id)):
            lines.append(f"### {m.resume_id} ({m.resume_class} / {m.difficulty})")
            if not m.parse_success:
                lines.append(f"- **FAILED**: `{m.error_type}` — {m.error_msg}")
            else:
                lines.append(f"- Sections detected: `{', '.join(m.detected_sections) or 'none'}`")
                lines.append(f"- SDR: `{m.sdr}` | SFDR: `{m.sfdr}`")
                lines.append(
                    f"- Skill precision: `{m.skill_precision}` | "
                    f"recall: `{m.skill_recall}` | F1: `{m.skill_f1}`"
                )
                lines.append(f"- TQS: `{m.tqs}` — {m.tqs_details}")
                lines.append(f"- Education accuracy: `{m.edu_accuracy}`")
                lines.append(f"- Experience accuracy: `{m.exp_accuracy}`")
                if m.missed_skills:
                    lines.append(f"- **Missed skills:** {', '.join(m.missed_skills)}")
                if m.false_positive_skills:
                    lines.append(
                        f"- **False positive skills (first 10):** "
                        f"{', '.join(m.false_positive_skills[:10])}"
                    )
                if m.known_issues:
                    lines.append(f"- _Known issues:_ {m.known_issues}")
            lines.append("")

    return "\n".join(lines)


def render_json(metrics: list[MetricSet], agg: dict) -> str:
    return json.dumps({"aggregated": agg, "per_resume": [asdict(m) for m in metrics]}, indent=2)


# ─────────────────── CLI ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate the resume parser against a labelled fixture corpus."
    )
    ap.add_argument("--resume-dir", type=Path, default=Path("test_resumes"),
                    help="Root directory of test resume files")
    ap.add_argument("--ground-truth-dir", type=Path, default=Path("test_resumes/ground_truth"),
                    help="Root directory of ground truth JSON files")
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="Directory containing skills_master.csv")
    ap.add_argument("--class", dest="resume_class", default=None,
                    help="Evaluate only this class (e.g. single_column)")
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write report to this file instead of stdout")
    ap.add_argument("--verbose", action="store_true",
                    help="Include per-resume detail section in report")
    args = ap.parse_args()

    print("Loading skill alias map...", file=sys.stderr)
    alias_map = load_alias_map(args.data_dir)
    print(f"  {len(alias_map):,} aliases loaded.", file=sys.stderr)

    print("Discovering corpus...", file=sys.stderr)
    records = discover_corpus(args.resume_dir, args.ground_truth_dir, args.resume_class)
    if not records:
        print(f"[ERROR] No resume fixtures found in {args.resume_dir}", file=sys.stderr)
        print(
            "Run first: python tests/fixtures/generate_fixtures.py",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  {len(records)} resume(s) found.", file=sys.stderr)

    print("Parsing...", file=sys.stderr)
    attempts = []
    for rec in records:
        label = f"  {rec.gt.resume_class}/{rec.path.name}"
        print(label, file=sys.stderr, end=" ... ")
        attempt = run_parse(rec)
        status = "OK" if attempt.success else f"FAILED ({attempt.error_type})"
        print(f"{status} [{attempt.latency_ms:.0f}ms]", file=sys.stderr)
        attempts.append(attempt)

    print("Computing metrics...", file=sys.stderr)
    all_metrics = [evaluate_one(a, alias_map) for a in attempts]

    print("Aggregating...", file=sys.stderr)
    agg = aggregate(all_metrics)

    report = (
        render_json(all_metrics, agg)
        if args.format == "json"
        else render_markdown(all_metrics, agg, verbose=args.verbose)
    )

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.output}", file=sys.stderr)
    else:
        safe = report.encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe)


if __name__ == "__main__":
    main()
