#!/usr/bin/env python3
"""
Skill Corpus Validation Framework
===================================
Measures the quality of the skill normalization pipeline against annotated benchmarks.
Standalone — no database, no server, no external dependencies beyond stdlib.

Metrics:
  M1  Alias Coverage Rate         — completeness and FP-risk scan per skill
  M2  Normalization Accuracy      — Tier 1 probe pass rate (T1a / T1b / T1c / T1d)
  M3  False Positive Rate         — Tier 3 adversarial sentence pass rate
  M4  False Negative Rate         — Tier 2 resume phrase recall
  M5  Alias Collision Report      — silent last-writer-wins detection
  M6  Seed Integrity              — seed_db.py ROLE_SKILL_PROFILES × skills_master.csv

Usage:
    python scripts/validate_corpus.py
    python scripts/validate_corpus.py --verbose
    python scripts/validate_corpus.py --format json --output corpus_report.json

Exit code 0 = all probes passed and seed is clean.
Exit code 1 = normalization failures or broken seed refs found.
"""

import argparse
import ast
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = ROOT / "tests" / "corpus"
SEED_DB_PATH = ROOT / "scripts" / "seed_db.py"


# ─────────────────────────────────────────────────────────────────────────────
# Boundary pattern
# ─────────────────────────────────────────────────────────────────────────────

def _make_boundary_pattern(alias: str) -> str:
    r"""
    Word-boundary-aware regex for alias matching.

    \b fails when the alias starts or ends with a non-word char (c++, c#, f#,
    .net c#). Use negative lookarounds in those cases.

      c++   → \bc\+\+(?!\w)
      c#    → \bc\#(?!\w)
      .net  → (?<!\w)\.net\b
    """
    if not alias:
        return re.escape(alias)
    escaped = re.escape(alias)
    prefix = r"\b" if alias[0].isalnum() or alias[0] == "_" else r"(?<!\w)"
    suffix = r"\b" if alias[-1].isalnum() or alias[-1] == "_" else r"(?!\w)"
    return prefix + escaped + suffix


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillEntry:
    canonical: str
    display: str
    category: str
    subcategory: str
    aliases: list = field(default_factory=list)


def load_skills(data_dir: Path = DATA_DIR) -> list:
    rows = []
    with open(data_dir / "skills_master.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            canonical = row["canonical_name"].strip()
            aliases = [a.strip() for a in (row.get("aliases") or "").split("|") if a.strip()]
            rows.append(SkillEntry(
                canonical=canonical,
                display=row["display_name"].strip(),
                category=row["category"].strip(),
                subcategory=row["subcategory"].strip(),
                aliases=aliases,
            ))
    return rows


def build_alias_map(skills: list) -> tuple:
    """
    Returns (alias_map, collisions).

    alias_map  : {alias_lower: canonical}  — last writer wins on collision
    collisions : [{alias, claimed_by: [canonical, ...]}]
    """
    raw: dict = {}
    for skill in skills:
        tokens = [skill.canonical, skill.display] + skill.aliases
        for tok in tokens:
            key = tok.strip().lower()
            if key:
                raw.setdefault(key, []).append(skill.canonical)

    alias_map = {k: v[-1] for k, v in raw.items()}
    collisions = [
        {"alias": k, "claimed_by": list(dict.fromkeys(v))}
        for k, v in raw.items()
        if len(set(v)) > 1
    ]
    return alias_map, collisions


def scan_text(text: str, alias_map: dict) -> set:
    """Return set of canonical names found in text using smart boundary matching."""
    if not text or not alias_map:
        return set()
    text_lower = text.lower()
    found: set = set()
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if not alias:
            continue
        if re.search(_make_boundary_pattern(alias), text_lower):
            found.add(alias_map[alias])
    return found


# ─────────────────────────────────────────────────────────────────────────────
# M1 — Alias Coverage Rate
# ─────────────────────────────────────────────────────────────────────────────

_HIGH_FP_RISK = {
    "go", "r", "c", "echo", "spring", "salt", "falcon", "ruby",
    "ray", "spark", "storm", "blade", "torch", "arch", "atlas",
}


def check_m1(skills: list) -> dict:
    total_aliases = sum(len(s.aliases) for s in skills)
    avg = round(total_aliases / len(skills), 2) if skills else 0.0

    few = [s.canonical for s in skills if len(s.aliases) < 2]

    fp_risk = []
    for skill in skills:
        risk_tokens = [skill.canonical] + skill.aliases
        risky = [t for t in risk_tokens if t.lower() in _HIGH_FP_RISK]
        if risky:
            fp_risk.append({"canonical": skill.canonical, "risky_tokens": risky})

    nonword_boundaries = []
    for skill in skills:
        for alias in skill.aliases:
            if alias and not (alias[-1].isalnum() or alias[-1] == "_"):
                nonword_boundaries.append({"canonical": skill.canonical, "alias": alias})

    return {
        "total_skills": len(skills),
        "total_aliases": total_aliases,
        "avg_aliases_per_skill": avg,
        "skills_with_fewer_than_2_aliases": few,
        "high_fp_risk_entries": fp_risk,
        "aliases_with_nonword_end": nonword_boundaries,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M2 — Normalization Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def _diagnose(inp: str, expected: str, skills: list, alias_map: dict) -> str:
    canonical_set = {s.canonical for s in skills}
    if expected not in canonical_set:
        return "UNKNOWN_CANONICAL_IN_PROBE"
    skill = next((s for s in skills if s.canonical == expected), None)
    if not skill:
        return "MISSING_ALIAS"
    all_toks = [skill.canonical.lower(), skill.display.lower()] + [a.lower() for a in skill.aliases]
    if not any(tok in inp.lower() for tok in all_toks):
        return "MISSING_ALIAS"
    return "BOUNDARY_BUG"


def check_m2(alias_map: dict, skills: list) -> dict:
    tier1_path = CORPUS_DIR / "tier1_normalization_probes.json"
    if not tier1_path.exists():
        return {"error": "tier1_normalization_probes.json not found", "probes": []}

    with open(tier1_path, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for probe in data.get("probes", []):
        inp = probe["input"]
        expected = probe["expected"]
        got = scan_text(inp, alias_map)
        passed = expected in got
        results.append({
            "subtier": probe.get("subtier", "T1?"),
            "input": inp,
            "expected": expected,
            "got": sorted(got),
            "passed": passed,
            "root_cause": None if passed else _diagnose(inp, expected, skills, alias_map),
            "description": probe.get("description", ""),
        })

    by_subtier: dict = {}
    for r in results:
        st = r["subtier"]
        by_subtier.setdefault(st, {"pass": 0, "fail": 0, "failures": []})
        if r["passed"]:
            by_subtier[st]["pass"] += 1
        else:
            by_subtier[st]["fail"] += 1
            by_subtier[st]["failures"].append(r)

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    return {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "accuracy": round(passed_count / total, 4) if total else 1.0,
        "by_subtier": by_subtier,
        "all_probes": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M3 — False Positive Rate (adversarial)
# ─────────────────────────────────────────────────────────────────────────────

def check_m3(alias_map: dict) -> dict:
    tier3_path = CORPUS_DIR / "tier3_adversarial.json"
    if not tier3_path.exists():
        return {"error": "tier3_adversarial.json not found", "probes": []}

    with open(tier3_path, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for probe in data.get("probes", []):
        text = probe["text"]
        forbidden = set(probe.get("forbidden_canonicals", []))
        severity = probe.get("severity", "critical")
        extracted = scan_text(text, alias_map)
        violations = extracted & forbidden if forbidden else set()
        results.append({
            "text": text,
            "description": probe.get("description", ""),
            "severity": severity,
            "forbidden": sorted(forbidden),
            "extracted": sorted(extracted),
            "violations": sorted(violations),
            "clean": len(violations) == 0,
        })

    critical = [r for r in results if r["severity"] == "critical"]
    known_risk = [r for r in results if r["severity"] == "known_risk"]

    fp_counts: dict = {}
    for r in results:
        for v in r["violations"]:
            fp_counts[v] = fp_counts.get(v, 0) + 1

    return {
        "total": len(results),
        "critical_total": len(critical),
        "critical_clean": sum(1 for r in critical if r["clean"]),
        "known_risk_total": len(known_risk),
        "known_risk_clean": sum(1 for r in known_risk if r["clean"]),
        "worst_offenders": sorted(fp_counts.items(), key=lambda x: x[1], reverse=True),
        "probes": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M4 — False Negative Rate (resume phrase recall)
# ─────────────────────────────────────────────────────────────────────────────

def check_m4(alias_map: dict) -> dict:
    tier2_path = CORPUS_DIR / "tier2_resume_phrases.json"
    if not tier2_path.exists():
        return {"error": "tier2_resume_phrases.json not found", "phrases": []}

    with open(tier2_path, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for phrase in data.get("phrases", []):
        text = phrase["text"]
        expected = set(phrase["expected_canonicals"])
        got = scan_text(text, alias_map)
        matched = expected & got
        missed = expected - got
        recall = round(len(matched) / len(expected), 4) if expected else 1.0
        results.append({
            "text": text,
            "description": phrase.get("description", ""),
            "expected": sorted(expected),
            "got": sorted(got),
            "missed": sorted(missed),
            "recall": recall,
        })

    total_expected = sum(len(r["expected"]) for r in results)
    total_missed = sum(len(r["missed"]) for r in results)
    overall_recall = round(1.0 - total_missed / total_expected, 4) if total_expected else 1.0

    miss_counts: dict = {}
    for r in results:
        for m in r["missed"]:
            miss_counts[m] = miss_counts.get(m, 0) + 1

    return {
        "total_phrases": len(results),
        "total_expected_extractions": total_expected,
        "total_missed": total_missed,
        "overall_recall": overall_recall,
        "fnr": round(total_missed / total_expected, 4) if total_expected else 0.0,
        "most_missed_skills": sorted(miss_counts.items(), key=lambda x: x[1], reverse=True),
        "phrases": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M5 — Alias Collisions
# ─────────────────────────────────────────────────────────────────────────────

def check_m5(collisions: list) -> dict:
    return {
        "total_collisions": len(collisions),
        "collisions": collisions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M6 — Seed Integrity
# ─────────────────────────────────────────────────────────────────────────────

def _extract_role_skill_refs(seed_db_path: Path) -> set:
    """
    Parse ROLE_SKILL_PROFILES from seed_db.py using the AST module.
    No import, no side effects — safe to call without a running DB.
    """
    try:
        source = seed_db_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "ROLE_SKILL_PROFILES":
                        refs: set = set()
                        if isinstance(node.value, ast.Dict):
                            for list_node in node.value.values:
                                if isinstance(list_node, ast.List):
                                    for elt in list_node.elts:
                                        if isinstance(elt, ast.Tuple) and elt.elts:
                                            first = elt.elts[0]
                                            if isinstance(first, ast.Constant):
                                                refs.add(first.value)
                        return refs
    except Exception as exc:
        print(f"[WARN] Could not parse seed_db.py: {exc}", file=sys.stderr)
    return set()


def check_m6(skills: list) -> dict:
    canonical_set = {s.canonical for s in skills}
    refs = _extract_role_skill_refs(SEED_DB_PATH)
    broken = sorted(r for r in refs if r not in canonical_set)
    return {
        "total_skill_refs": len(refs),
        "broken_refs": broken,
        "broken_count": len(broken),
        "valid_count": len(refs) - len(broken),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _pct(num: int, den: int) -> str:
    return "N/A" if den == 0 else f"{100 * num / den:.1f}%"


def _generate_action_items(results: dict) -> list:
    actions = []
    m2 = results.get("m2", {})
    m3 = results.get("m3", {})
    m4 = results.get("m4", {})
    m5 = results.get("m5", {})
    m6 = results.get("m6", {})

    for subtier in ("T1b", "T1c", "T1a", "T1d"):
        fails = m2.get("by_subtier", {}).get(subtier, {}).get("fail", 0)
        if fails:
            label = {
                "T1b": "variant spellings (missing aliases in CSV)",
                "T1c": "punctuation aliases (boundary bug in _alias_scan / scan_text_for_skills)",
                "T1a": "round-trip probes (canonical/display name mismatch)",
                "T1d": "in-context probes (alias or boundary issue in realistic text)",
            }[subtier]
            actions.append(("P0", f"{fails} {subtier} probe(s) fail — {label}"))

    broken = m6.get("broken_count", 0)
    if broken:
        refs = m6.get("broken_refs", [])[:5]
        suffix = "..." if m6.get("broken_count", 0) > 5 else ""
        actions.append(("P1", f"{broken} broken skill ref(s) in seed_db.py ROLE_SKILL_PROFILES: "
                       f"{', '.join(refs)}{suffix}"))

    few = len(results.get("m1", {}).get("skills_with_fewer_than_2_aliases", []))
    if few:
        actions.append(("P2", f"{few} skill(s) have < 2 explicit aliases — increase coverage"))

    crit_fp = m3.get("critical_total", 0) - m3.get("critical_clean", 0)
    if crit_fp:
        offenders = m3.get("worst_offenders", [])[:3]
        actions.append(("P2", f"{crit_fp} critical adversarial probe(s) trigger false positives: "
                       f"{', '.join(c for c, _ in offenders)}"))

    known_fp = m3.get("known_risk_total", 0) - m3.get("known_risk_clean", 0)
    if known_fp:
        actions.append(("P3", f"{known_fp} known-risk English-word false positive(s) — "
                       "require context-aware matching to fix (not alias-scan fixable)"))

    if m5.get("total_collisions"):
        actions.append(("P3", f"{m5['total_collisions']} alias collision(s) — "
                       "last-writer wins silently, audit and deduplicate"))

    fnr = m4.get("fnr", 0)
    if fnr > 0.1:
        missed = m4.get("most_missed_skills", [])[:5]
        actions.append(("P2", f"Overall recall {_pct(int((1 - fnr) * 100), 100)} on phrase corpus — "
                       f"most missed: {', '.join(c for c, _ in missed)}"))

    return sorted(actions, key=lambda x: x[0])


def render_markdown(results: dict, verbose: bool = False) -> str:
    lines: list = []
    w = lines.append

    m1 = results["m1"]
    w("# Skill Corpus Validation Report")
    w(f"\nCorpus: `data/skills_master.csv`  ")
    w(f"Skills: **{m1['total_skills']}** | "
      f"Total aliases: **{m1['total_aliases']}** | "
      f"Avg per skill: **{m1['avg_aliases_per_skill']}**")

    # M1
    w("\n## M1 — Alias Coverage")
    w(f"- Skills with < 2 explicit aliases: **{len(m1['skills_with_fewer_than_2_aliases'])}**")
    w(f"- High FP-risk entries (common English words): **{len(m1['high_fp_risk_entries'])}**")
    w(f"- Aliases with non-word-char endings (now handled via lookaheads): **{len(m1['aliases_with_nonword_end'])}**")
    if verbose and m1["high_fp_risk_entries"]:
        w("\n  High-risk canonicals:")
        for e in m1["high_fp_risk_entries"]:
            w(f"  - `{e['canonical']}` — risky tokens: {e['risky_tokens']}")

    # M2
    w("\n## M2 — Normalization Accuracy")
    m2 = results["m2"]
    if "error" in m2:
        w(f"  ⚠ {m2['error']}")
    else:
        w(f"Overall: **{m2['passed']}/{m2['total']}** ({_pct(m2['passed'], m2['total'])})")
        for st in ["T1a", "T1b", "T1c", "T1d"]:
            data = m2.get("by_subtier", {}).get(st)
            if data is None:
                continue
            total_st = data["pass"] + data["fail"]
            status = "✓" if data["fail"] == 0 else "✗"
            w(f"- {st}: {status} {data['pass']}/{total_st}")
            if verbose and data["failures"]:
                for failure in data["failures"]:
                    w(f"  - FAIL `{failure['input']!r}` → expected `{failure['expected']}` | "
                      f"root_cause: `{failure['root_cause']}`")

    # M3
    w("\n## M3 — False Positive Rate (Adversarial)")
    m3 = results["m3"]
    if "error" in m3:
        w(f"  ⚠ {m3['error']}")
    else:
        w(f"Critical probes clean: **{m3['critical_clean']}/{m3['critical_total']}**  ")
        w(f"Known-risk probes clean: **{m3['known_risk_clean']}/{m3['known_risk_total']}** *(documented limitations)*")
        if m3["worst_offenders"]:
            w(f"Worst offenders: {', '.join(f'`{c}` ({n}×)' for c, n in m3['worst_offenders'][:5])}")
        if verbose:
            for p in m3["probes"]:
                if not p["clean"]:
                    sev = p["severity"]
                    w(f"  - [{sev.upper()}] `{p['text'][:70]}` → violations: {p['violations']}")

    # M4
    w("\n## M4 — False Negative Rate (Resume Phrase Recall)")
    m4 = results["m4"]
    if "error" in m4:
        w(f"  ⚠ {m4['error']}")
    else:
        w(f"Overall recall: **{_pct(m4['total_expected_extractions'] - m4['total_missed'], m4['total_expected_extractions'])}**  ")
        w(f"FNR: **{_pct(m4['total_missed'], m4['total_expected_extractions'])}** "
          f"({m4['total_missed']}/{m4['total_expected_extractions']} extractions missed)")
        if m4["most_missed_skills"]:
            w(f"Most missed: {', '.join(f'`{c}` ({n}×)' for c, n in m4['most_missed_skills'][:8])}")
        if verbose:
            for p in m4["phrases"]:
                if p["missed"]:
                    w(f"  - MISS `{p['text'][:70]}` → missed: {p['missed']}")

    # M5
    w("\n## M5 — Alias Collisions")
    m5 = results["m5"]
    if m5["total_collisions"] == 0:
        w("No collisions detected. ✓")
    else:
        w(f"Total collisions: **{m5['total_collisions']}**")
        if verbose:
            for c in m5["collisions"][:15]:
                w(f"  - `{c['alias']}` claimed by: {c['claimed_by']}")

    # M6
    w("\n## M6 — Seed Integrity (`seed_db.py` × `skills_master.csv`)")
    m6 = results["m6"]
    w(f"Refs in ROLE_SKILL_PROFILES: **{m6['total_skill_refs']}** | "
      f"Valid: **{m6['valid_count']}** | Broken: **{m6['broken_count']}**")
    if m6["broken_refs"]:
        w(f"\nBroken refs (skill canonical not in CSV):")
        for ref in m6["broken_refs"]:
            w(f"  - `{ref}`")

    # Action items
    w("\n## Action Items")
    actions = _generate_action_items(results)
    if not actions:
        w("No action items — corpus is healthy. ✓")
    else:
        for priority, msg in actions:
            w(f"- **{priority}**: {msg}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(data_dir: Path = DATA_DIR) -> dict:
    """Run all checks and return the full results dict."""
    skills = load_skills(data_dir)
    alias_map, collisions = build_alias_map(skills)
    return {
        "m1": check_m1(skills),
        "m2": check_m2(alias_map, skills),
        "m3": check_m3(alias_map),
        "m4": check_m4(alias_map),
        "m5": check_m5(collisions),
        "m6": check_m6(skills),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate skill corpus quality.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to data/ directory")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", help="Write report to file instead of stdout")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-probe details")
    args = parser.parse_args()

    results = run(Path(args.data_dir))

    if args.format == "json":
        report = json.dumps(
            results,
            indent=2,
            default=lambda x: sorted(x) if isinstance(x, (set, frozenset)) else str(x),
        )
    else:
        report = render_markdown(results, verbose=args.verbose)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report.encode("ascii", errors="replace").decode("ascii"))

    m2 = results.get("m2", {})
    m6 = results.get("m6", {})
    m3 = results.get("m3", {})
    crit_fp = m3.get("critical_total", 0) - m3.get("critical_clean", 0)
    if m2.get("failed", 0) > 0 or m6.get("broken_count", 0) > 0 or crit_fp > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
