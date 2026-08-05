"""
Recruiter intelligence layer.

Derives hiring signals, evidence annotations, and project recommendations
from already-computed gap analysis data. Zero additional DB queries or LLM calls.

Used by pages.py to enrich the results page context before rendering.
"""

from __future__ import annotations

from src.services.gap_analyzer import GAP_MULTIPLIERS, _maturity_bonus

_PROJECT_CLUSTERS: list[dict] = [
    {
        "skill_tags": {
            "Redis", "Celery", "RabbitMQ", "Apache Kafka", "Amazon SQS",
        },
        "title": "Async Job Processing Platform",
        "description": (
            "Build a distributed task queue with reliable job processing, "
            "configurable retry logic, dead-letter handling, and a live monitoring dashboard."
        ),
        "deliverables": [
            "Producer/consumer architecture with pluggable broker",
            "Dead-letter queue with automatic retry on transient failure",
            "Real-time queue depth and throughput monitoring endpoint",
        ],
        "portfolio_artifact": "GitHub repo with Docker Compose + load-test benchmark",
        "difficulty": "Intermediate",
        "estimated_hours": 20,
        "recruiter_value": "High",
    },
    {
        "skill_tags": {
            "FastAPI", "Flask", "Django", "PostgreSQL", "Express.js", "Spring Boot",
        },
        "title": "Production-Grade REST API",
        "description": (
            "Build a full-featured REST API with rate limiting, pagination, "
            "structured error handling, and auto-generated OpenAPI documentation."
        ),
        "deliverables": [
            "Fully documented API spec with Swagger UI",
            "Rate limiting, input validation, and structured error responses",
            "Deployed with automated health checks and CI/CD",
        ],
        "portfolio_artifact": "Live API URL + GitHub repo with README",
        "difficulty": "Intermediate",
        "estimated_hours": 16,
        "recruiter_value": "High",
    },
    {
        "skill_tags": {
            "Docker", "Kubernetes", "Terraform", "Amazon Web Services",
            "Google Cloud Platform", "Microsoft Azure", "GitHub Actions",
        },
        "title": "Container Orchestration Pipeline",
        "description": (
            "Containerize a real project, write Kubernetes manifests with auto-scaling, "
            "and automate deployment via a CI/CD pipeline with rollback."
        ),
        "deliverables": [
            "Optimized Dockerfiles with multi-stage builds",
            "Kubernetes manifests: Deployment, Service, HPA",
            "GitHub Actions pipeline with automated rollback on failure",
        ],
        "portfolio_artifact": "GitHub repo with k8s/ directory + architecture diagram",
        "difficulty": "Advanced",
        "estimated_hours": 25,
        "recruiter_value": "High",
    },
    {
        "skill_tags": {
            "PyTorch", "TensorFlow", "Hugging Face Transformers",
            "Transformers Architecture", "Large Language Models",
        },
        "title": "ML Pipeline: Fine-Tune to Deployment",
        "description": (
            "Fine-tune a model on a custom dataset, evaluate it systematically "
            "against baselines, and expose it via a low-latency inference API."
        ),
        "deliverables": [
            "Training script with experiment tracking (MLflow or W&B)",
            "Evaluation report with benchmark comparisons",
            "Inference API with latency and throughput benchmarks",
        ],
        "portfolio_artifact": "HuggingFace model card + GitHub repo",
        "difficulty": "Advanced",
        "estimated_hours": 30,
        "recruiter_value": "High",
    },
    {
        "skill_tags": {
            "pytest", "Selenium", "Playwright", "Cypress", "Jest",
        },
        "title": "QA Automation Suite",
        "description": (
            "Write a complete E2E and integration test suite for an existing project, "
            "with CI integration, coverage reporting, and a coverage gate."
        ),
        "deliverables": [
            "E2E test suite covering all critical user flows",
            "Integration tests with mocked external dependencies",
            "CI pipeline with 80%+ coverage gate required to merge",
        ],
        "portfolio_artifact": "GitHub Actions green CI badge + coverage report screenshot",
        "difficulty": "Intermediate",
        "estimated_hours": 18,
        "recruiter_value": "High",
    },
    {
        "skill_tags": {
            "React", "Vue.js", "Next.js", "TypeScript", "Angular", "Svelte",
        },
        "title": "Full-Stack Web Application",
        "description": (
            "Build a production-ready web app with a modern framework, "
            "state management, API integration, and public deployment."
        ),
        "deliverables": [
            "Accessible component library with consistent design system",
            "API integration with optimistic updates and error states",
            "Publicly deployed with Lighthouse performance score",
        ],
        "portfolio_artifact": "Live demo URL + GitHub repo",
        "difficulty": "Intermediate",
        "estimated_hours": 22,
        "recruiter_value": "High",
    },
    {
        "skill_tags": {
            "Retrieval-Augmented Generation", "Embeddings", "LangChain",
            "LlamaIndex", "Hugging Face Transformers",
        },
        "title": "RAG-Powered Semantic Search",
        "description": (
            "Build a retrieval-augmented generation system with document ingestion, "
            "semantic search, re-ranking, and a grounded chat interface."
        ),
        "deliverables": [
            "Document ingestion pipeline with chunking strategy",
            "Vector search with relevance scoring and re-ranking",
            "Chat interface with source attribution and hallucination guards",
        ],
        "portfolio_artifact": "Live demo + GitHub with architecture diagram",
        "difficulty": "Advanced",
        "estimated_hours": 28,
        "recruiter_value": "High",
    },
    {
        "skill_tags": {
            "Apache Spark", "Airflow DAGs", "dbt",
        },
        "title": "End-to-End Data Pipeline",
        "description": (
            "Build a batch or streaming data pipeline with ingestion, "
            "transformation, quality checks, and a reporting layer."
        ),
        "deliverables": [
            "Ingestion layer with configurable source connectors",
            "Transformation layer with data quality assertions",
            "Aggregated reporting table with automated freshness check",
        ],
        "portfolio_artifact": "GitHub repo with pipeline DAG diagram + sample data",
        "difficulty": "Advanced",
        "estimated_hours": 26,
        "recruiter_value": "High",
    },
]


def _make_project_rationale(
    overlap: set[str],
    foundation: set[str],
    profile: dict,
) -> str:
    """Generate a one-line explanation of why this project fits this candidate."""
    gaps = sorted(overlap)[:2]
    built_on = sorted(foundation)[:2]
    exp = profile.get("experience_level", "mid")

    if built_on and gaps:
        on_str = " and ".join(built_on)
        gap_str = " and ".join(gaps)
        return (
            f"Leverages your {on_str} background to close the {gap_str} gap — "
            "high ROI given your existing foundation."
        )
    if gaps:
        gap_str = " and ".join(gaps)
        if exp == "junior":
            return f"Builds hands-on depth in {gap_str} — strong portfolio signal for early-career roles."
        return f"Directly addresses {gap_str} — a priority gap for this role."
    return "Strong portfolio signal that complements your current skill set."


def compute_project_recommendations(
    missing_skills: list[dict],
    matched_skills: list[dict] | None = None,
    candidate_profile: dict | None = None,
) -> list[dict]:
    """
    Match missing skills to buildable portfolio projects.

    Scoring:
      gap_score    (0.7 weight) — how many cluster skills are in the candidate's gap list
      synergy_score (0.3 weight) — how many cluster skills the candidate already has
    Difficulty penalty: advanced projects are de-weighted for junior-level candidates.
    Returns top 3 by combined score, each with a why_this_project explanation.
    """
    if not missing_skills:
        return []

    profile = candidate_profile or {}
    exp_level = profile.get("experience_level", "mid")

    missing_names = {s["display_name"] for s in missing_skills}
    matched_names = {s["display_name"] for s in (matched_skills or [])}

    results: list[dict] = []
    for cluster in _PROJECT_CLUSTERS:
        overlap = cluster["skill_tags"] & missing_names
        if not overlap:
            continue
        foundation = cluster["skill_tags"] & matched_names
        gap_score     = len(overlap) / len(cluster["skill_tags"])
        synergy_score = len(foundation) / len(cluster["skill_tags"])
        total_score   = gap_score * 0.7 + synergy_score * 0.3

        # De-weight advanced projects for junior candidates
        if exp_level == "junior" and cluster["difficulty"] == "Advanced":
            total_score *= 0.65

        rec = {k: v for k, v in cluster.items() if k != "skill_tags"}
        rec["matched_gaps"]      = sorted(overlap)
        rec["foundation_skills"] = sorted(foundation)
        rec["relevance_score"]   = round(total_score, 3)
        rec["why_this_project"]  = _make_project_rationale(overlap, foundation, profile)
        results.append(rec)

    _value = {"High": 2, "Medium": 1, "Low": 0}
    results.sort(
        key=lambda x: (x["relevance_score"], _value.get(x["recruiter_value"], 0)),
        reverse=True,
    )
    return results[:3]


def compute_recruiter_summary(
    gap_data: dict,
    readiness_score: int,
    candidate_profile: dict | None = None,
) -> dict:
    """
    Hiring-signal summary answering four recruiter questions:
      1. Why is this candidate valuable?
      2. Why would I interview them?
      3. What concerns remain?
      4. What role level fits best?

    All existing return keys preserved for backward compatibility.
    Pure computation — zero DB queries or LLM calls.
    """
    profile = candidate_profile or {}
    breakdown: dict       = gap_data.get("category_breakdown", {})
    matched_skills: list  = gap_data.get("matched_skills", [])
    missing_skills: list  = gap_data.get("missing_skills", [])

    adjacent_gap_count = profile.get("adjacent_gap_count", 0)
    open_source        = profile.get("open_source_signals", False)
    mentorship         = profile.get("mentorship_signals", False)
    leadership         = profile.get("leadership_signals", False)
    core_strengths     = profile.get("core_strengths", [])
    primary_domain     = profile.get("primary_domain", "")
    exp                = profile.get("experience_level", "junior")

    # Index matched skills by category
    matched_by_cat: dict[str, list[str]] = {}
    for s in matched_skills:
        cat = s.get("category") or "Other"
        matched_by_cat.setdefault(cat, []).append(s["display_name"])

    # ── Strengths: categories with ≥ 60% coverage and ≥ 2 matched skills ────
    strengths: list[dict] = []
    for cat, data in breakdown.items():
        total   = data.get("total", 0)
        matched = data.get("matched", 0)
        if total < 2:
            continue
        pct = matched / total
        if pct >= 0.60:
            evidence = matched_by_cat.get(cat, [])[:5]
            if evidence:
                strengths.append({
                    "label":        cat,
                    "coverage_pct": round(pct * 100),
                    "evidence":     evidence,
                })
    strengths.sort(key=lambda x: x["coverage_pct"], reverse=True)

    # ── Concerns: categories with < 35% coverage and ≥ 3 required skills ────
    concerns: list[dict] = []
    for cat, data in breakdown.items():
        total   = data.get("total", 0)
        matched = data.get("matched", 0)
        missing = data.get("missing", 0)
        if total < 3:
            continue
        pct = matched / total
        if pct < 0.35:
            concerns.append({
                "label":         cat,
                "coverage_pct":  round(pct * 100),
                "missing_count": missing,
            })
    concerns.sort(key=lambda x: x["coverage_pct"])

    critical_gaps     = sum(1 for s in missing_skills if s.get("importance_score", 0) >= 0.7)
    foundational_gaps = sum(1 for s in missing_skills if s.get("gap_type") == "foundational")
    n_strengths       = len(strengths)

    # ── Verdict tier ─────────────────────────────────────────────────────────
    if readiness_score >= 75:
        verdict, color = "Strong Interview Candidate", "green"
    elif readiness_score >= 55:
        verdict, color = "Promising Candidate", "yellow"
    elif readiness_score >= 35:
        verdict, color = "Needs Additional Experience", "orange"
    else:
        verdict, color = "Significant Skill Gap", "red"

    # ── Reasoning (backward-compat single-string field) ──────────────────────
    if readiness_score >= 75:
        reasoning = (
            f"Covers {readiness_score}% of role requirements with strong technical depth "
            f"across {n_strengths} domain{'s' if n_strengths != 1 else ''}."
        )
    elif readiness_score >= 55:
        reasoning = (
            f"Solid foundation with {critical_gaps} high-priority "
            f"gap{'s' if critical_gaps != 1 else ''} addressable through focused preparation."
        )
    elif readiness_score >= 35:
        reasoning = (
            f"Core skills present but {critical_gaps} critical gap{'s' if critical_gaps != 1 else ''} "
            "identified. Recommend 3–6 months of structured preparation."
        )
    else:
        reasoning = "Multiple foundational skills missing. The roadmap below provides a structured path forward."

    if adjacent_gap_count > 0 and readiness_score >= 35:
        reasoning += (
            f" {adjacent_gap_count} apparent gap{'s' if adjacent_gap_count != 1 else ''} "
            "are adjacent expertise or transferable — real ramp-up is shorter than the raw count suggests."
        )
    if open_source:
        reasoning += " Open-source contributions demonstrate collaborative, production-grade coding habits."

    # ── Suggested level — profile-driven, not readiness-driven ──────────────
    if exp == "senior":
        suggested_level = "Mid-Level"
    elif exp == "mid":
        suggested_level = "Strong Junior"
    else:
        suggested_level = "Junior / Entry"

    if exp == "senior" and readiness_score >= 72 and critical_gaps == 0:
        suggested_level = "Mid–Senior"

    if readiness_score < 30:
        suggested_level = "Early Career / Developing"

    # ── Value proposition: why is this candidate valuable? ───────────────────
    vp_parts: list[str] = []
    if core_strengths:
        top = core_strengths[:3]
        vp_parts.append(f"Demonstrated depth in {', '.join(top)}")
    if open_source:
        vp_parts.append("open-source contributions showing production-grade coding habits")
    if mentorship:
        vp_parts.append("mentorship background indicating communication ability above typical junior level")
    if adjacent_gap_count > 0:
        vp_parts.append(
            f"{adjacent_gap_count} apparent gap{'s' if adjacent_gap_count != 1 else ''} "
            "that are actually adjacent expertise — shorter real ramp-up than skill count implies"
        )
    if primary_domain:
        vp_parts.append(f"primary domain strength in {primary_domain}")

    value_proposition = (
        ". ".join(s[0].upper() + s[1:] for s in vp_parts) + "."
        if vp_parts
        else "Candidate profile available — review matched skills for specific signals."
    )

    # ── Interview rationale: why bring them in? ──────────────────────────────
    ir_parts: list[str] = []
    if readiness_score >= 55:
        ir_parts.append(
            f"Covers {readiness_score}% of role requirements — above the screening threshold for a junior hire"
        )
    if open_source and leadership:
        ir_parts.append("combination of OSS work and leadership signals is uncommon at this experience level")
    elif open_source:
        ir_parts.append("open-source track record provides verifiable code quality evidence")
    if adjacent_gap_count > 0:
        ir_parts.append(
            "several 'missing' skills are adjacent to confirmed expertise — "
            "worth verifying in-person whether the gap is real or a labelling artefact"
        )
    if foundational_gaps == 0 and critical_gaps <= 1:
        ir_parts.append("no foundational gaps — candidate has the conceptual base to grow into the role")

    interview_rationale = (
        ". ".join(s[0].upper() + s[1:] for s in ir_parts) + "."
        if ir_parts
        else (
            "Candidate meets baseline screening criteria — worth a phone screen to validate depth."
            if readiness_score >= 35
            else "Consider only if the pipeline is thin or the role can accommodate a longer ramp."
        )
    )

    # ── Remaining concerns: honest, not flattering ────────────────────────────
    concern_parts: list[str] = []
    if foundational_gaps > 0:
        fgaps = [s["display_name"] for s in missing_skills if s.get("gap_type") == "foundational"][:3]
        concern_parts.append(
            f"Foundational gap{'s' if foundational_gaps != 1 else ''} in "
            f"{', '.join(fgaps)} — these take months to build, not weeks"
        )
    real_critical = max(0, critical_gaps - adjacent_gap_count)
    if real_critical > 0:
        concern_parts.append(
            f"{real_critical} high-importance skill{'s' if real_critical != 1 else ''} "
            "genuinely missing (not adjacent) — active preparation required"
        )
    if readiness_score < 50 and not open_source:
        concern_parts.append(
            "No verifiable portfolio or open-source evidence — harder to confirm depth of claimed skills"
        )
    if not concern_parts:
        concern_parts.append("No critical concerns identified at this stage")

    remaining_concerns = ". ".join(s[0].upper() + s[1:] for s in concern_parts) + "."

    # ── Interview focus — foundational gaps first (highest risk) ─────────────
    interview_focus: list[str] = []
    sorted_missing = sorted(missing_skills, key=lambda x: (
        {"foundational": 0, "domain": 1, "tooling": 2}.get(x.get("gap_type", "tooling"), 1),
        -x.get("importance_score", 0),
    ))
    for s in sorted_missing[:5]:
        gap_status = s.get("gap_status", "missing")
        via_skill  = s.get("via_skill")
        gap_type   = s.get("gap_type", "tooling")
        importance = s.get("importance_score", 0)

        if gap_status == "adjacent_expertise" and via_skill:
            interview_focus.append(
                f"How would you apply your {via_skill} experience to ramp up on {s['display_name']}?"
            )
        elif gap_status == "transferable" and via_skill:
            interview_focus.append(
                f"Walk through how {via_skill} prepared you for {s['display_name']} — what's still new?"
            )
        elif gap_type == "foundational":
            interview_focus.append(
                f"Depth of {s['display_name']} fundamentals — explain trade-offs, not just API usage"
            )
        elif importance >= 0.5:
            interview_focus.append(
                f"Depth of experience with {s['display_name']} — production usage and edge cases"
            )

    if not interview_focus and missing_skills:
        interview_focus.append(
            f"Explore breadth of {missing_skills[0]['display_name']} and adjacent technologies"
        )

    return {
        # ── Existing keys (backward compat) ──────────────────────────────────
        "strengths":           strengths[:3],
        "concerns":            concerns[:3],
        "verdict":             verdict,
        "verdict_color":       color,
        "reasoning":           reasoning,
        "suggested_level":     suggested_level,
        "interview_focus":     interview_focus[:3],
        # ── New structured fields ─────────────────────────────────────────────
        "value_proposition":   value_proposition,
        "interview_rationale": interview_rationale,
        "remaining_concerns":  remaining_concerns,
        "foundational_gaps":   foundational_gaps,
        "critical_gaps":       critical_gaps,
    }


def compute_role_fit_ranking(
    user_skill_ids: set[int],
    all_role_profiles: list[dict],
    current_role_id: int | None = None,
    user_skill_names: set[str] | None = None,
) -> dict:
    """
    Cross-role fit comparison from a single pre-fetched bulk query.
    Returns ranked list + three application-readiness buckets.

    Fit % = sum(effective importance of candidate's skills) / sum(all role skill importances).
    Adjacent and transferable skills receive partial credit (50% and 30% respectively)
    so a Python/FastAPI developer is not scored the same as someone with no backend experience
    when evaluated against a Java-centric Backend Engineer profile.

    Buckets:
      apply_now        fit >= 65% — competitive application today
      need_experience  40–64%    — gap closable in 1–3 months
      farther_away     < 40%     — significant preparation needed
    """
    from collections import defaultdict
    from src.services.skill_graph import build_user_skill_index, classify_via_graph

    # Build adjacency index from the user's full skill name set (if provided)
    user_skill_index = build_user_skill_index(list(user_skill_names or set()))

    role_meta: dict[int, dict] = {}
    # skill entry: {imp: float, name: str}
    role_skills: dict[int, dict[int, dict]] = defaultdict(dict)

    for row in all_role_profiles:
        rid = row["role_id"]
        if rid not in role_meta:
            role_meta[rid] = {
                "display_name": row["display_name"],
                "domain":       row["domain"],
            }
        role_skills[rid][row["skill_id"]] = {
            "imp":  row["importance_score"],
            "name": row.get("skill_name", ""),
        }

    results: list[dict] = []
    for rid, skills in role_skills.items():
        total_imp = sum(v["imp"] for v in skills.values())
        if total_imp == 0:
            continue

        matched_imp = 0.0
        matched_skill_dicts: list[dict] = []
        for sid, v in skills.items():
            if sid in user_skill_ids:
                matched_imp += v["imp"]
                if v["name"]:
                    matched_skill_dicts.append({"display_name": v["name"]})
            elif user_skill_index and v["name"]:
                # Apply the same adjacency multipliers as analyze_gap() so the
                # ranking score is consistent with the main readiness score.
                graph_status, _ = classify_via_graph(v["name"], user_skill_index)
                if graph_status == "adjacent_expertise":
                    matched_imp += v["imp"] * GAP_MULTIPLIERS["adjacent_expertise"]
                elif graph_status == "transferable":
                    matched_imp += v["imp"] * GAP_MULTIPLIERS["transferable"]

        # Maturity bonus: same capped signal used by analyze_gap() so the ranking
        # score matches the Executive Summary score for the current role.
        maturity_pts = _maturity_bonus(matched_skill_dicts)
        fit_pct = min(100, round((matched_imp / total_imp) * 100) + maturity_pts)

        results.append({
            "role_id":      rid,
            "display_name": role_meta[rid]["display_name"],
            "domain":       role_meta[rid]["domain"],
            "fit_pct":      fit_pct,
            "is_current":   rid == current_role_id,
        })

    results.sort(key=lambda x: (-x["fit_pct"], x["display_name"]))

    apply_now       = [r for r in results if r["fit_pct"] >= 65 and not r["is_current"]][:5]
    need_experience = [r for r in results if 40 <= r["fit_pct"] < 65 and not r["is_current"]][:5]
    farther_away    = [r for r in results if r["fit_pct"] < 40 and not r["is_current"]][:3]
    ranked          = [r for r in results if r["fit_pct"] >= 40][:8]

    return {
        "apply_now":       apply_now,
        "need_experience": need_experience,
        "farther_away":    farther_away,
        "ranked":          ranked,
    }


_WEEKS_BY_GAP_TYPE: dict[str, int] = {
    "tooling":      2,   # specific tool — learn by building one project
    "domain":       5,   # specialisation area — needs structured study + project
    "foundational": 9,   # CS/ML theory — courses + practice + consolidation
}

# Conservative: if adjacent/transferable skills already give partial credit,
# the remaining learning gap is smaller — reflect that in the weeks estimate.
_WEEKS_ADJACENCY_DISCOUNT: dict[str, float] = {
    "adjacent_expertise": 0.35,   # ~35% of the full ramp-up time needed
    "transferable":       0.60,
    "related":            0.80,
    "partial_match":      0.90,
    "missing":            1.00,
}


def compute_shortest_path(
    missing_skills: list[dict],
    total_importance: float,
    adjusted_readiness_score: int,
    target_threshold: int = 65,
) -> dict:
    """
    Find the minimum skill set that would push adjusted readiness past target_threshold.

    Selection strategy:
      1. Exclude skills that already have adjacent/transferable partial credit
         when computing net gain (they contribute less because they already score partial).
      2. Sort: highest net gain first, then prefer tooling over domain/foundational
         (quicker wins; roadmap should close tooling gaps before theory gaps).
      3. Accumulate until threshold is crossed.

    All time estimates are conservative — err long rather than short.
    Returns None-equivalent dict with already_there=True when no gap exists.
    """
    if adjusted_readiness_score >= target_threshold or total_importance <= 0:
        return {
            "already_there":   True,
            "current_score":   adjusted_readiness_score,
            "target_score":    target_threshold,
            "roi_skills":      [],
            "estimated_weeks": 0,
            "projected_score": adjusted_readiness_score,
        }

    _GAP_TYPE_ORDER = {"tooling": 0, "domain": 1, "foundational": 2}

    candidates = []
    for s in missing_skills:
        gap_status  = s.get("gap_status", "missing")
        gap_type    = s.get("gap_type", "tooling")
        imp         = s.get("importance_score", 0.0)
        full_impact = (imp / total_importance) * 100
        # Net gain = what we still need to learn (partial credit already applied)
        current_mult = GAP_MULTIPLIERS.get(gap_status, 0.0)
        net_gain_pts = full_impact * (1.0 - current_mult)
        # Weeks = base estimate × adjacency discount
        base_weeks   = _WEEKS_BY_GAP_TYPE.get(gap_type, 2)
        adj_discount = _WEEKS_ADJACENCY_DISCOUNT.get(gap_status, 1.0)
        est_weeks    = max(1, round(base_weeks * adj_discount))

        if net_gain_pts < 0.5:   # trivial impact — skip
            continue

        candidates.append({
            "display_name": s["display_name"],
            "gap_type":     gap_type,
            "gap_status":   gap_status,
            "net_gain_pts": round(net_gain_pts, 1),
            "est_weeks":    est_weeks,
            "_order":       _GAP_TYPE_ORDER.get(gap_type, 1),
        })

    # Sort by net gain per week invested — maximises speed to threshold.
    # A 14pt gain in 2 weeks (7pt/wk) beats a 16pt gain in 9 weeks (1.8pt/wk).
    # Tie-break by type order so tooling edges out foundational when equal.
    candidates.sort(key=lambda x: (-x["net_gain_pts"] / x["est_weeks"], x["_order"]))

    selected: list[dict] = []
    accumulated_gain = 0.0
    total_weeks      = 0

    for c in candidates:
        if accumulated_gain >= (target_threshold - adjusted_readiness_score):
            break
        accumulated_gain += c["net_gain_pts"]
        total_weeks      += c["est_weeks"]
        selected.append({
            "display_name": c["display_name"],
            "gap_type":     c["gap_type"],
            "impact_pts":   c["net_gain_pts"],
            "weeks":        c["est_weeks"],
        })

    projected_score = min(100, adjusted_readiness_score + round(accumulated_gain))

    return {
        "already_there":   False,
        "current_score":   adjusted_readiness_score,
        "target_score":    target_threshold,
        "roi_skills":      selected[:6],
        "estimated_weeks": total_weeks,
        "projected_score": projected_score,
    }


def compute_evidence_map(gap_data: dict) -> dict[int, dict]:
    """
    For each missing skill, produce a why/related_found/graph_note annotation.
    Uses same-category matched skills as evidence of adjacent competency.
    graph_note uses gap_status + via_skill (now embedded in missing_skills) to
    produce a recruiter-grade shortcut note ("Your PyTorch experience transfers to TensorFlow").
    Keyed by integer skill_id for direct Jinja2 access.
    """
    matched_by_cat: dict[str, list[str]] = {}
    for s in gap_data.get("matched_skills", []):
        cat = s.get("category") or "Other"
        matched_by_cat.setdefault(cat, []).append(s["display_name"])

    evidence: dict[int, dict] = {}
    for skill in gap_data.get("missing_skills", []):
        skill_id:   int   = skill["skill_id"]
        cat:        str   = skill.get("category") or "Other"
        importance: float = skill.get("importance_score", 0.0)
        gap_status: str   = skill.get("gap_status", "missing")
        via_skill:  str | None = skill.get("via_skill")

        related_found = matched_by_cat.get(cat, [])[:3]

        if importance >= 0.7:
            why = (
                f"Consistently required for {cat} roles — "
                "appears in the top demanded skills for this position."
            )
        elif importance >= 0.4:
            why = f"Commonly expected in {cat} positions at mid-to-senior levels."
        else:
            why = f"Beneficial for {cat} work — adds versatility to your profile."

        # Graph note: explain the shortcut a recruiter would mentally make
        graph_note: str | None = None
        if gap_status == "adjacent_expertise" and via_skill:
            graph_note = (
                f"Your {via_skill} experience is directly transferable — "
                "this is adjacent expertise, not a from-scratch gap."
            )
        elif gap_status == "transferable" and via_skill:
            graph_note = (
                f"Your {via_skill} background covers the conceptual foundation — "
                "a natural next step in your skill progression."
            )
        elif gap_status == "related":
            graph_note = "You already work in this domain — this specific tool fills a targeted gap within familiar territory."

        evidence[skill_id] = {
            "why":           why,
            "related_found": related_found,
            "graph_note":    graph_note,
        }

    return evidence
