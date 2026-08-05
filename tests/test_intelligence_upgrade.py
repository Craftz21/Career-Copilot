"""Integration tests for the career intelligence upgrade (8 phases)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.candidate_profiler import build_candidate_profile
from src.services.recruiter import compute_recruiter_summary, compute_shortest_path
from src.services.skill_graph import classify_gap_type
from src.services.soft_skill_inferencer import build_skill_evidence

errors = []


def check(cond, msg):
    if not cond:
        errors.append(f"FAIL  {msg}")


# ── Phase 1: Level thresholds ─────────────────────────────────────────────────

p_junior = build_candidate_profile(
    [{"display_name": f"S{i}", "category": "ML", "importance_score": 0.8} for i in range(10)],
    [],
    [
        {"skill": "Leadership", "confidence": 0.75},
        {"skill": "Mentorship", "confidence": 0.72},
        {"skill": "Open Source Contribution", "confidence": 0.72},
    ],
)
p_mid = build_candidate_profile(
    [{"display_name": f"S{i}", "category": "ML", "importance_score": 0.8} for i in range(14)],
    [],
    [
        {"skill": "Leadership", "confidence": 0.75},
        {"skill": "Open Source Contribution", "confidence": 0.72},
    ],
)
p_senior = build_candidate_profile(
    [{"display_name": f"S{i}", "category": "ML", "importance_score": 0.8} for i in range(20)],
    [],
    [
        {"skill": "Leadership", "confidence": 0.75},
        {"skill": "Mentorship", "confidence": 0.72},
        {"skill": "Open Source Contribution", "confidence": 0.72},
    ],
)

check(p_junior["experience_level"] == "junior",
      f"10 matched + all 3 signals should = junior, got {p_junior['experience_level']}")
check(p_mid["experience_level"] == "mid",
      f"14 matched + leadership + OSS should = mid, got {p_mid['experience_level']}")
check(p_senior["experience_level"] == "senior",
      f"20 matched + all 3 signals should = senior, got {p_senior['experience_level']}")

print(f"P1  junior={p_junior['experience_level']}  mid={p_mid['experience_level']}  senior={p_senior['experience_level']}")

# ── Phase 1: suggested_level ──────────────────────────────────────────────────

g = {"category_breakdown": {}, "matched_skills": [], "missing_skills": []}
sl_j   = compute_recruiter_summary(g, 55, p_junior)["suggested_level"]
sl_m   = compute_recruiter_summary(g, 55, p_mid)["suggested_level"]
sl_s55 = compute_recruiter_summary(g, 55, p_senior)["suggested_level"]
sl_s72 = compute_recruiter_summary(g, 72, p_senior)["suggested_level"]
sl_low = compute_recruiter_summary(g, 25, p_senior)["suggested_level"]

check(sl_j   == "Junior / Entry",           f"junior@55%  should=Junior/Entry got {sl_j}")
check(sl_m   == "Strong Junior",            f"mid@55%     should=Strong Junior got {sl_m}")
check(sl_s55 == "Mid-Level",                f"senior@55%  should=Mid-Level got {sl_s55}")
# senior@72%: critical_gaps=0 is required for Mid-Senior. g has no missing_skills so critical_gaps=0.
check(sl_s72 == "Mid–Senior",               f"senior@72%+0crits should=Mid-Senior got {sl_s72}")
check(sl_low == "Early Career / Developing", f"senior@25%  should=Early Career/Developing got {sl_low}")

print(f"P1  jr@55%={sl_j}  mid@55%={sl_m}  sr@55%={sl_s55}  sr@72%={sl_s72}  sr@25%={sl_low}")

# ── Phase 5: gap type classification ─────────────────────────────────────────

type_cases = [
    ("Algorithms",                  "foundational"),
    ("Deep Learning",               "foundational"),
    ("System Design",               "foundational"),
    ("Distributed Systems",         "foundational"),
    ("Natural Language Processing", "domain"),
    ("Computer Vision",             "domain"),
    ("MLflow",                      "tooling"),
    ("Amazon Web Services",         "tooling"),
    ("Docker",                      "tooling"),
    ("DVC",                         "tooling"),
]
for name, expected in type_cases:
    got = classify_gap_type(name)
    check(got == expected, f"classify_gap_type({name!r}) expected {expected!r} got {got!r}")

print("P5  " + " | ".join(f"{n}={classify_gap_type(n)}" for n, _ in type_cases[:4]))

# ── Phase 4: shortest path ────────────────────────────────────────────────────

missing_sp = [
    {"display_name": "MLflow",       "importance_score": 0.7,  "gap_status": "missing",            "gap_type": "tooling"},
    {"display_name": "Deep Learning","importance_score": 0.8,  "gap_status": "missing",            "gap_type": "foundational"},
    {"display_name": "AWS",          "importance_score": 0.6,  "gap_status": "missing",            "gap_type": "tooling"},
    {"display_name": "TensorFlow",   "importance_score": 0.75, "gap_status": "adjacent_expertise", "gap_type": "tooling"},
]
sp = compute_shortest_path(missing_sp, total_importance=5.0, adjusted_readiness_score=50, target_threshold=65)

check(not sp["already_there"],         "already_there should be False")
check(sp["current_score"] == 50,       f"current_score should=50 got {sp['current_score']}")
check(sp["target_score"] == 65,        f"target_score should=65 got {sp['target_score']}")
check(len(sp["roi_skills"]) > 0,       "roi_skills should not be empty")
check(sp["estimated_weeks"] > 0,       "estimated_weeks should be > 0")
check(sp["projected_score"] > 50,      f"projected_score should > 50 got {sp['projected_score']}")

# Tooling gaps should sort before foundational (quicker wins first)
types_in_order = [s["gap_type"] for s in sp["roi_skills"]]
first_foundational = next((i for i, t in enumerate(types_in_order) if t == "foundational"), len(types_in_order))
first_tooling      = next((i for i, t in enumerate(types_in_order) if t == "tooling"), len(types_in_order))
check(first_tooling <= first_foundational, "tooling should appear before foundational in roi_skills")

print(f"P4  roi={[(s['display_name'], s['gap_type'], s['weeks']) for s in sp['roi_skills']]}")
print(f"P4  weeks={sp['estimated_weeks']}  projected={sp['projected_score']}%")

# Already-there path
sp2 = compute_shortest_path([], 0.0, 70, 65)
check(sp2["already_there"], "should be already_there when score >= threshold")
print(f"P4  already_there at 70% = {sp2['already_there']}")

# ── Phase 6: recruiter summary new fields ────────────────────────────────────

gd = {
    "category_breakdown": {"ML": {"total": 5, "matched": 3, "missing": 2}},
    "matched_skills": [{"display_name": "PyTorch", "category": "ML", "importance_score": 0.9}],
    "missing_skills": [
        {"display_name": "TensorFlow",    "importance_score": 0.8,  "gap_status": "adjacent_expertise",
         "gap_type": "tooling",      "via_skill": "PyTorch", "skill_id": 1},
        {"display_name": "Deep Learning", "importance_score": 0.85, "gap_status": "missing",
         "gap_type": "foundational", "via_skill": None,      "skill_id": 2},
    ],
}
s2 = compute_recruiter_summary(gd, 60, p_mid)

for key in ("value_proposition", "interview_rationale", "remaining_concerns", "foundational_gaps", "critical_gaps"):
    check(key in s2, f"summary missing key {key!r}")

check(s2.get("foundational_gaps") == 1, f"foundational_gaps should=1 got {s2.get('foundational_gaps')}")
check(len(s2.get("value_proposition", "")) > 20,   "value_proposition too short")
check(len(s2.get("interview_rationale", "")) > 20, "interview_rationale too short")
check(len(s2.get("remaining_concerns", "")) > 20,  "remaining_concerns too short")

# Interview focus should mention foundational gap before tooling (foundational = highest risk)
focus = s2.get("interview_focus", [])
print(f"P6  verdict={s2['verdict']}  level={s2['suggested_level']}  foundational_gaps={s2['foundational_gaps']}")
print(f"P6  vp:  {s2['value_proposition'][:70]}")
print(f"P6  ir:  {s2['interview_rationale'][:70]}")
print(f"P6  rc:  {s2['remaining_concerns'][:70]}")
print(f"P6  if:  {focus}")

# ── Phase 7: evidence quality ─────────────────────────────────────────────────

sections = {
    "projects": (
        "Built a digital twin simulator using PyTorch RNNs that reduced forecasting error by 40%. "
        "Deployed the model API to HuggingFace Spaces for public demo."
    ),
    "skills": "Python PyTorch TensorFlow scikit-learn",
}
matched_ev = [{"skill_id": 1, "display_name": "PyTorch", "category": "ML", "importance_score": 0.9}]
ev = build_skill_evidence(sections, matched_ev)

check(1 in ev,                          "skill_id 1 should be in evidence")
check(ev.get(1, {}).get("confidence") == "High",
      f"projects section should give High confidence, got {ev.get(1, {}).get('confidence')}")
check("impact" in ev.get(1, {}),       "impact field should be present")
check("section" in ev.get(1, {}),      "section field should be present")
check("snippet" in ev.get(1, {}),      "snippet field should be present")

if 1 in ev:
    print(f"P7  section={ev[1]['section']}  confidence={ev[1]['confidence']}")
    print(f"P7  impact:  {ev[1]['impact']}")
    print(f"P7  snippet: {ev[1]['snippet'][:70]}")

# Skill not present in text → not in evidence (no false positives)
ev2 = build_skill_evidence(sections, [{"skill_id": 99, "display_name": "Kubernetes", "category": "DevOps", "importance_score": 0.8}])
check(99 not in ev2, "Kubernetes not mentioned in text — should not appear in evidence")
print(f"P7  Kubernetes not in text, in evidence: {99 in ev2} (expected: False)")

# ── Result ────────────────────────────────────────────────────────────────────

print()
if errors:
    print(f"FAILED ({len(errors)} errors):")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("All integration tests passed.")
