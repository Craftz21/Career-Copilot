"""
Roadmap generation via Groq LLM with structured output, prompt versioning,
and template-based fallback (never shows 500 to the user).

Flow:
  1. Check DB cache — short-lived connection, released before LLM call.
  2. Call Groq (llama-3.3-70b-versatile). Validate JSON with Pydantic.
  3. On validation failure → retry once with explicit format reminder.
  4. On second failure → generate template-based roadmap from learning_resources.
  5. Store result in DB cache so identical role+gap combos are free on repeat.

Performance optimisations (O3 + O6):
  O3 — max_tokens reduced 4096 → 2000 (roadmaps rarely exceed 1500 tokens;
       this cuts average generation time by ~30–50%).
  O6 — generate_roadmap() no longer accepts a db Session. It manages its own
       short-lived connections so the Celery task does not hold an open
       transaction during the 4–15 s LLM call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel, ValidationError
from sqlalchemy import text

from src.config import get_settings
from src.database import get_db
from src.services.resource_catalog import enrich_roadmap_resources

log = structlog.get_logger(__name__)
settings = get_settings()

_PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"
_PROMPT_CACHE: dict[str, str] = {}  # version → template string


# ---------------------------------------------------------------------------
# Pydantic schema for LLM output validation
# ---------------------------------------------------------------------------

class RoadmapResource(BaseModel):
    title: str
    platform: str
    url: Optional[str] = None
    estimated_hours: Optional[int] = None
    type: str  # "course" | "project" | "documentation" | "book"


class RoadmapWeek(BaseModel):
    week_number: int
    focus: str
    skills: list[str]
    tasks: list[str]
    resources: list[RoadmapResource]
    learn: Optional[list[str]] = None
    build: Optional[str] = None
    deploy: Optional[str] = None
    showcase: Optional[str] = None


class RoadmapPhase(BaseModel):
    phase_number: int
    title: str
    goal: str
    duration_weeks: int
    builds_on: list[str] = []
    closes_gaps: list[str] = []
    actions: list[str]
    deliverable: str
    deploy_target: Optional[str] = None
    resources: list[RoadmapResource]


class RoadmapOutput(BaseModel):
    title: str
    total_duration_weeks: int
    summary: str
    # Phase-based (v2 prompt) — preferred
    phases: Optional[list[RoadmapPhase]] = None
    # Week-based (template fallback / v1 prompt) — backward compat
    weeks: Optional[list[RoadmapWeek]] = None
    success_metrics: list[str]


# ---------------------------------------------------------------------------
# Public interface  (O6: no db parameter — manages connections internally)
# ---------------------------------------------------------------------------

def generate_roadmap(
    session_id: str,
    role_display_name: str,
    gap_data: dict[str, Any],
    duration: str,
    prompt_version: str = "v2",
) -> dict[str, Any]:
    """
    Generate and return a structured learning roadmap.

    Manages its own DB connections (two short-lived contexts) so that the
    caller does not hold an open transaction during the LLM call.

    Returns:
        {
            "content": RoadmapOutput dict,
            "model_used": str,
            "generation_ms": int,
            "cache_hit": bool,
            "prompt_version": str,
        }
    """
    t0 = time.monotonic()
    cache_key = _build_cache_key(gap_data, role_display_name, duration)

    # ── Phase A: cache check (fast, short-lived connection) ──────────────────
    with get_db() as db:
        cached = _check_cache(cache_key, db)

    if cached:
        log.info("roadmap_cache_hit", session_id=str(session_id))
        return {**cached, "cache_hit": True, "generation_ms": 0}

    # ── Phase B: LLM call (no DB connection held) ────────────────────────────
    prompt = _build_prompt(role_display_name, gap_data, duration, prompt_version)
    content, model_used = _call_llm_with_retry(prompt)

    if content is None:
        log.warning("roadmap_llm_failed_using_template", session_id=str(session_id))
        # Template fallback needs DB for learning_resources — open fresh connection
        with get_db() as db:
            content = _template_fallback(role_display_name, gap_data, duration, db)
        model_used = "template_fallback"

    # Enrich per-week resources with curated catalog entries (covers both paths).
    content = enrich_roadmap_resources(content)

    generation_ms = int((time.monotonic() - t0) * 1000)

    # Embed cache key into content dict so the caller can persist it
    _store_cache(cache_key, content)

    log.info(
        "roadmap_generated",
        session_id=str(session_id),
        model=model_used,
        generation_ms=generation_ms,
    )

    return {
        "content": content,
        "model_used": model_used,
        "generation_ms": generation_ms,
        "cache_hit": False,
        "prompt_version": prompt_version,
    }


# ---------------------------------------------------------------------------
# LLM call (O3: max_tokens 4096 → 2000)
# ---------------------------------------------------------------------------

def _call_llm_with_retry(prompt: str) -> tuple[Optional[dict], str]:
    """Returns (parsed_content_dict | None, model_name_str)."""
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    model = settings.llm_model

    system_msg = (
        "You are a senior career coach and technical curriculum designer. "
        "Always respond ONLY with valid JSON matching the exact schema provided. "
        "No markdown fences, no commentary — raw JSON only."
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=3000,  # Phase-based output (4 phases + resources) needs more room
            )
            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = RoadmapOutput.model_validate_json(raw)
            return parsed.model_dump(), model

        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            log.warning("roadmap_llm_parse_error", attempt=attempt, error=str(exc))
            if attempt == 0:
                prompt += (
                    "\n\nIMPORTANT: Your previous response had a JSON format error. "
                    "Return ONLY raw JSON with no markdown or extra text."
                )
            continue
        except Exception as exc:
            log.error("roadmap_llm_error", attempt=attempt, error=str(exc))
            break

    return None, model


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _load_prompt_template(version: str) -> str:
    if version not in _PROMPT_CACHE:
        path = _PROMPT_DIR / f"roadmap_{version}.txt"
        if not path.exists():
            path = _PROMPT_DIR / "roadmap_v1.txt"
        _PROMPT_CACHE[version] = path.read_text(encoding="utf-8")
    return _PROMPT_CACHE[version]


def _build_prompt(
    role: str,
    gap_data: dict,
    duration: str,
    version: str,
) -> str:
    template = _load_prompt_template(version)

    missing = gap_data.get("missing_skills", [])[:15]
    matched = gap_data.get("matched_skills", [])[:10]
    # Use adjusted score (same number shown in Executive Summary) so the
    # LLM-generated roadmap text is consistent with what the user sees on screen.
    readiness = gap_data.get("adjusted_readiness_score", gap_data.get("readiness_score", 0))

    missing_list = "\n".join(
        f"- {s['display_name']} (importance: {s['importance_score']:.2f})"
        for s in missing
    )
    matched_list = "\n".join(f"- {s['display_name']}" for s in matched)

    return template.format(
        role=role,
        duration=duration,
        readiness_score=readiness,
        missing_skills=missing_list or "None identified",
        matched_skills=matched_list or "None identified",
    )


# ---------------------------------------------------------------------------
# Template fallback (when LLM fails)
# ---------------------------------------------------------------------------

def _template_fallback(
    role: str,
    gap_data: dict,
    duration: str,
    db,
) -> dict:
    """
    Build a basic roadmap from pre-seeded learning_resources.
    Returns a dict matching RoadmapOutput shape.
    """
    missing = gap_data.get("missing_skills", [])[:12]
    duration_weeks = _parse_duration_weeks(duration)

    per_week = max(1, len(missing) // max(duration_weeks, 1))
    weeks = []
    for i in range(duration_weeks):
        chunk = missing[i * per_week : (i + 1) * per_week]
        if not chunk and i > 0:
            break
        skill_ids = [s["skill_id"] for s in chunk]
        resources = _fetch_resources(skill_ids, db) if skill_ids else []
        tasks: list[str] = []
        for s in chunk:
            tasks.append(f"Study {s['display_name']} fundamentals and core concepts")
            tasks.append(f"Build a hands-on project applying {s['display_name']}")
        weeks.append({
            "week_number": i + 1,
            "focus": ", ".join(s["display_name"] for s in chunk) or "Review & Practice",
            "skills": [s["display_name"] for s in chunk],
            "tasks": tasks[:6],
            "resources": resources,
        })

    return {
        "title": f"{role} Skill Development Roadmap",
        "total_duration_weeks": duration_weeks,
        "summary": (
            f"A {duration} roadmap to bridge your skill gaps for the {role} role. "
            f"Your current readiness is {gap_data.get('adjusted_readiness_score', gap_data.get('readiness_score', 0))}%."
        ),
        "weeks": weeks,
        "success_metrics": [
            "Complete all listed resources",
            "Build at least one project using the learned skills",
            "Contribute to an open-source project in your target stack",
        ],
    }


def _fetch_resources(skill_ids: list[int], db) -> list[dict]:
    if not skill_ids:
        return []
    rows = db.execute(
        text(
            """
            SELECT title, platform, resource_type, estimated_hours
            FROM learning_resources
            WHERE skill_id = ANY(:ids)
            LIMIT 5
            """
        ),
        {"ids": skill_ids},
    ).fetchall()
    return [
        {
            "title": r.title,
            "platform": r.platform,
            "url": None,
            "estimated_hours": r.estimated_hours,
            "type": r.resource_type,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _build_cache_key(gap_data: dict, role: str, duration: str) -> str:
    top_missing = sorted(
        [s["skill_id"] for s in gap_data.get("missing_skills", [])[:10]]
    )
    return f"{role}::{duration}::{','.join(str(s) for s in top_missing)}"


def _check_cache(cache_key: str, db) -> Optional[dict]:
    row = db.execute(
        text(
            """
            SELECT content, model_used, prompt_version
            FROM roadmaps
            WHERE content->>'_cache_key' = :key
            LIMIT 1
            """
        ),
        {"key": cache_key},
    ).first()
    if row:
        return {
            "content": row.content,
            "model_used": row.model_used,
            "prompt_version": row.prompt_version,
        }
    return None


def _store_cache(cache_key: str, content: dict) -> None:
    """Embed the cache key inside the content dict (persisted by the Celery task)."""
    content["_cache_key"] = cache_key


def _parse_duration_weeks(duration: str) -> int:
    """Convert duration string like '4 weeks', '3 months', '6 months' → int weeks."""
    duration_lower = duration.lower()
    if "month" in duration_lower:
        try:
            months = int("".join(filter(str.isdigit, duration_lower)))
            return months * 4
        except ValueError:
            return 12
    if "week" in duration_lower:
        try:
            return int("".join(filter(str.isdigit, duration_lower)))
        except ValueError:
            return 8
    return 8  # default
