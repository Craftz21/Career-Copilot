"""
Soft skill inference and resume evidence extraction.

Two public functions:
  infer_soft_skills(sections)          — detect behavioral soft skills from resume text
  build_skill_evidence(sections, skills) — locate matched skills in resume sections

Both are pure functions: no DB, no LLM, no side effects.
Called from the Celery task (to cache pre-wipe) and from pages.py (fresh fallback).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Inference rules — behavioral signals → soft skill inferences
# ---------------------------------------------------------------------------

_INFERENCE_RULES: list[dict] = [
    {
        "patterns": [
            r"collaborat\w+\s+with",
            r"worked\s+(?:alongside|with)\s+\w+",
            r"team\s+of\s+\d+",
            r"cross[\s-]functional",
            r"pair\s+programming",
        ],
        "infers": [("Collaboration", 0.70), ("Teamwork", 0.65)],
    },
    {
        "patterns": [
            # Require strong, unambiguous leadership verbs — not merely "led" followed by anything
            r"\bled\s+(?:a\s+)?(?:team|project|initiative|development|design|effort|group)",
            r"\bmanaged\s+(?:a\s+)?(?:team|group|project|pipeline|squad)",
            r"\bteam\s+lead\b",
            r"\btech(?:nical)?\s+lead\b",
            r"\bspearheaded\b",
            r"\bowned\s+(?:the\s+)?(?:architecture|design|development|roadmap|delivery)",
            r"\bsupervised\s+\w+",
            r"\bdirected\s+\w+",
        ],
        "infers": [("Leadership", 0.75), ("Ownership", 0.68)],
    },
    {
        "patterns": [
            # User must be the one doing the mentoring (not being mentored)
            # "mentored junior engineers" ✓  / "collaborated with mentors" ✗
            r"\bmentor(?:ed|ing)\s+(?:junior|intern|new|student|team\s+member)",
            r"\bcoached?\s+(?:junior|intern|new|student)\s+\w+",
            r"\bonboarded?\s+(?:junior|intern|new)\s+\w+",
            r"\bguided\s+(?:junior|intern|new|student|team\s+member)",
            r"\btrained\s+(?:junior|intern|new)\s+\w+",
        ],
        "infers": [("Mentorship", 0.72)],
        # Leadership NOT inferred from mentorship alone — collaboration ≠ leadership
    },
    {
        "patterns": [
            r"present\w+\s+(?:to|results|findings)",
            r"demo\w+\s+to",
            r"\bstakeholder\b",
            r"communicated?\s+\w+\s+(?:to|with)",
            r"\bcoordinated\s+with\b",
        ],
        "infers": [("Communication", 0.70), ("Stakeholder Management", 0.65)],
    },
    {
        "patterns": [
            r"open[\s-]source",
            r"contributed?\s+to\s+\w+",
            r"pull\s+request",
            r"\bpr\b.*merged",
            r"github\.com/\w+",
        ],
        "infers": [("Open Source Contribution", 0.72), ("Collaboration", 0.60)],
    },
    {
        "patterns": [
            r"research\w+\s+(?:on|into|of)",
            r"analyzed?\s+\w+\s+data",
            r"\bhypothesis\b",
            r"experiment\w+\s+with",
            r"literature\s+review",
        ],
        "infers": [("Analytical Thinking", 0.68), ("Problem Solving", 0.65)],
    },
    {
        "patterns": [
            r"\bdeadline\b",
            r"shipped?\s+\w+\s+(?:in|within|on)\s+(?:time|schedule)",
            r"\bsprint\b",
            r"\bagile\b",
            r"\bscrum\b",
        ],
        "infers": [("Ownership", 0.68), ("Time Management", 0.65)],
    },
    {
        "patterns": [
            r"improved?\s+\w+\s+by\s+\d+",
            r"optimized?\s+\w+",
            r"reduced?\s+\w+\s+by",
            r"increased?\s+\w+\s+by",
            r"cut\s+\w+\s+(?:time|cost|latency)",
        ],
        "infers": [("Impact Orientation", 0.70), ("Problem Solving", 0.68)],
    },
    {
        "patterns": [
            r"self[\s-]taught",
            r"independently\s+\w+",
            r"self[\s-]directed",
            r"\bproactively\b",
        ],
        "infers": [("Self-Motivation", 0.70), ("Ownership", 0.65)],
    },
]

# Minimum confidence for a soft skill to be shown on the results page.
# Below this threshold the evidence is too weak to be useful and may be misleading.
_MIN_DISPLAY_CONFIDENCE: float = 0.65

# Sections where behavioral language is most likely
_PRIORITY_SECTION_KEYS = {"experience", "work_experience", "projects", "leadership", "activities", "achievements"}


def infer_soft_skills(sections: dict) -> list[dict]:
    """
    Detect soft skills from behavioral language in resume sections.

    Returns:
        [{"skill": str, "confidence": float, "evidence": list[str (snippet)]}]
        sorted by confidence descending.
    """
    if not sections:
        return []

    # Split into priority and secondary sections for evidence quality
    primary_text: list[tuple[str, str]] = []
    secondary_text: list[tuple[str, str]] = []
    for name, text in sections.items():
        if not isinstance(text, str) or name.startswith("_"):
            continue
        if any(k in name.lower() for k in _PRIORITY_SECTION_KEYS):
            primary_text.append((name, text))
        else:
            secondary_text.append((name, text))

    all_sections = primary_text + secondary_text

    skill_scores: dict[str, dict] = {}

    for rule in _INFERENCE_RULES:
        for pattern in rule["patterns"]:
            for section_name, section_text in all_sections:
                for match in re.finditer(pattern, section_text, re.IGNORECASE):
                    start = max(0, match.start() - 35)
                    end = min(len(section_text), match.end() + 70)
                    snippet = " ".join(section_text[start:end].split())

                    for skill_name, confidence in rule["infers"]:
                        entry = skill_scores.setdefault(skill_name, {
                            "skill": skill_name,
                            "confidence": confidence,
                            "evidence": [],
                        })
                        entry["confidence"] = max(entry["confidence"], confidence)
                        if snippet and snippet not in entry["evidence"] and len(snippet) > 12:
                            entry["evidence"].append(snippet)

    result = list(skill_scores.values())
    for entry in result:
        entry["evidence"] = entry["evidence"][:2]   # cap at 2 snippets per skill
    # Drop inferences below the confidence floor — they are more likely false positives
    # than genuine behavioral signals and will mislead the recruiter view.
    result = [e for e in result if e["confidence"] >= _MIN_DISPLAY_CONFIDENCE and e["evidence"]]
    result.sort(key=lambda x: x["confidence"], reverse=True)
    return result


# Section priority for evidence confidence scoring
_SECTION_CONFIDENCE: dict[str, str] = {
    "projects":         "High",
    "experience":       "High",
    "work_experience":  "High",
    "work experience":  "High",
    "research":         "High",
    "achievements":     "High",
    "education":        "Medium",
    "activities":       "Medium",
    "leadership":       "Medium",
    "skills":           "Low",
    "technical_skills": "Low",
    "certifications":   "Low",
}

# Action verbs that signal applied usage (not just listed)
_IMPACT_VERBS = re.compile(
    r"\b(built|developed|designed|implemented|deployed|created|led|architected|"
    r"reduced|improved|increased|optimised|optimized|achieved|delivered|shipped|"
    r"trained|fine-tuned|integrated|automated|scaled|maintained|contributed|"
    r"published|released|launched|migrated|refactored|wrote|authored)\b",
    re.IGNORECASE,
)


def _extract_impact(snippet: str) -> str | None:
    """Return a short impact phrase if the snippet contains a measurable result or action."""
    # Quantified result: "reduced latency by 40%", "improved accuracy to 92%"
    quant = re.search(
        r"(reduced?|improved?|increased?|achieved?|cut|boosted?)\s+\w[\w\s]{0,30}"
        r"(by\s+\d+[\w%]+|to\s+\d+[\w%]+|from\s+\d+[\w%]+)",
        snippet,
        re.IGNORECASE,
    )
    if quant:
        return quant.group(0).strip()[:80]
    # Applied action: "built X using Y", "deployed to Z"
    applied = re.search(
        r"(built|developed|deployed|designed|implemented|trained|created|shipped)\s+[\w\s,]+",
        snippet,
        re.IGNORECASE,
    )
    if applied:
        return applied.group(0).strip()[:80]
    return None


def build_skill_evidence(sections: dict, matched_skills: list[dict]) -> dict[int, dict]:
    """
    For each matched skill, find the best resume sentence containing its name.

    Ranking: Projects/Experience sections beat Skills sections; longer, action-verb-rich
    sentences beat bare mentions. Returns {skill_id: {section, snippet, confidence, impact}}.

    Skills matched via embedding only (no literal alias in text) are silently skipped —
    the chip still renders without an evidence panel.
    """
    if not sections or not matched_skills:
        return {}

    # Ordered sections: higher-confidence sections first
    _CONFIDENCE_ORDER = {"High": 0, "Medium": 1, "Low": 2}
    ordered_sections: list[tuple[str, str, str]] = []
    for name, text in sections.items():
        if not isinstance(text, str) or name.startswith("_"):
            continue
        conf = _SECTION_CONFIDENCE.get(name.lower().replace(" ", "_"), "Low")
        ordered_sections.append((name, text, conf))
    ordered_sections.sort(key=lambda x: _CONFIDENCE_ORDER.get(x[2], 2))

    evidence: dict[int, dict] = {}
    for skill in matched_skills:
        name_lower = skill["display_name"].lower()
        skill_id   = skill["skill_id"]
        best: dict | None = None

        # Build a word-boundary pattern so "git" does not match "github" or "digital".
        # Skills with spaces ("rest api") use a non-word-boundary check on the whole phrase.
        if " " in name_lower:
            # Multi-word skill: require exact phrase present in sentence
            skill_pat = re.compile(re.escape(name_lower), re.IGNORECASE)
        else:
            # Single-word skill: require word boundary to prevent substring false positives
            skill_pat = re.compile(r"\b" + re.escape(name_lower) + r"\b", re.IGNORECASE)

        for section_name, section_text, conf in ordered_sections:
            sentences = re.split(r"(?<=[.!?\n])\s+|\n{2,}", section_text)
            for sentence in sentences:
                s = sentence.strip()
                if len(s) < 20 or not skill_pat.search(s):
                    continue
                has_verb   = bool(_IMPACT_VERBS.search(s))
                word_count = len(s.split())

                # Score: prefer high-confidence section + action verbs + longer sentence
                score = (
                    (2 if conf == "High" else 1 if conf == "Medium" else 0)
                    + (2 if has_verb else 0)
                    + min(word_count, 20) // 5   # up to 4 bonus for length
                )

                if best is None or score > best["_score"]:
                    best = {
                        "section":    section_name.replace("_", " ").title(),
                        "snippet":    s[:200],
                        "confidence": conf,
                        "impact":     _extract_impact(s),
                        "_score":     score,
                    }

            # Stop searching sections once we've found a High-confidence hit
            if best and best["confidence"] == "High":
                break

        # Suppress evidence if the best match is too weak to be meaningful.
        # A score < 2 means: no action verb AND not from a high-confidence section —
        # the mention is likely incidental (e.g. skill listed in skills section only).
        if best and best["_score"] >= 2:
            best.pop("_score")
            evidence[skill_id] = best

    return evidence
