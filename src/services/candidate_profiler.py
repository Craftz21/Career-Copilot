"""
Candidate Reasoning Engine.

Builds a structured internal profile from already-computed gap analysis data
and soft skill inferences. Used by recruiter.py and pages.py to personalise
recommendations and scoring narratives.

Pure function: zero DB queries, zero LLM calls.
Operates only on data already produced by analyze_gap() and infer_soft_skills().
"""

from __future__ import annotations


def build_candidate_profile(
    matched_skills: list[dict],
    missing_skills: list[dict],
    soft_inferences: list[dict],
) -> dict:
    """
    Build an internal candidate reasoning profile.

    Returns:
        {
          "core_strengths":        list[str]   top matched skills by importance score
          "primary_domain":        str          dominant skill category by coverage
          "secondary_domains":     list[str]
          "experience_level":      str          "junior" | "mid" | "senior"
          "leadership_signals":    bool
          "mentorship_signals":    bool
          "open_source_signals":   bool
          "collaboration_signals": bool
          "inferred_soft_skills":  list[str]
          "transferable_domains":  list[str]    categories where user has skills AND gaps
          "skill_count":           int
          "adjacent_gap_count":    int          gaps that are adjacent/transferable (not truly missing)
        }
    """
    # Core strengths: highest-importance matched skills
    sorted_matched = sorted(
        matched_skills, key=lambda x: x.get("importance_score", 0.0), reverse=True
    )
    core_strengths = [s["display_name"] for s in sorted_matched[:6]]

    # Domain coverage
    cat_counts: dict[str, int] = {}
    for s in matched_skills:
        cat = s.get("category") or "Other"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    sorted_cats = sorted(cat_counts, key=lambda c: -cat_counts[c])
    primary_domain = sorted_cats[0] if sorted_cats else "General"
    secondary_domains = sorted_cats[1:3]

    # Soft skill signals
    soft_map = {s["skill"]: s.get("confidence", 0.0) for s in soft_inferences}
    leadership    = "Leadership" in soft_map or "Ownership" in soft_map
    mentorship    = "Mentorship" in soft_map
    open_source   = "Open Source Contribution" in soft_map
    collaboration = "Collaboration" in soft_map or "Teamwork" in soft_map

    # Experience level inference — conservative by design.
    # Resume behavioral signals do not confirm industry tenure. A student TA
    # triggers mentorship; a hackathon organiser triggers leadership. Both are
    # valid early-career signals but do not equal 3–5 years of industry work.
    # Thresholds are intentionally high to avoid inflating level estimates.
    n = len(matched_skills)
    if leadership and mentorship and open_source and n >= 20:
        experience_level = "senior"
    elif (
        (leadership and open_source and n >= 14)
        or (mentorship and open_source and n >= 14)
        or (leadership and mentorship and n >= 18)
    ):
        experience_level = "mid"
    else:
        experience_level = "junior"

    # Transferable domains: categories with both matches and gaps
    matched_cats = {s.get("category") or "Other" for s in matched_skills}
    missing_cats = {s.get("category") or "Other" for s in missing_skills}
    transferable_domains = sorted(matched_cats & missing_cats)

    # Count gaps that are actually adjacent or transferable — not truly missing
    adjacent_gap_count = sum(
        1 for s in missing_skills
        if s.get("gap_status") in ("adjacent_expertise", "transferable", "related")
    )

    return {
        "core_strengths":        core_strengths,
        "primary_domain":        primary_domain,
        "secondary_domains":     secondary_domains,
        "experience_level":      experience_level,
        "leadership_signals":    leadership,
        "mentorship_signals":    mentorship,
        "open_source_signals":   open_source,
        "collaboration_signals": collaboration,
        "inferred_soft_skills":  list(soft_map.keys()),
        "transferable_domains":  transferable_domains,
        "skill_count":           len(matched_skills),
        "adjacent_gap_count":    adjacent_gap_count,
    }
