"""
Role normalization: free-text role name → canonical role_id.

Three-stage lookup (deterministic → string → semantic):
  1. Exact match: canonical_name (snake_case), display_name, or any alias — case-insensitive.
  2. Fuzzy string similarity (difflib) against display names and aliases.
     Handles partial names ("Software Engineer" → "Backend Software Engineer"),
     typos, and casing variants. Runs before embeddings — no model load required.
  3. pgvector cosine similarity — last resort when string similarity is too low.

Thresholds:
  Exact match          → confidence 1.0,  always wins
  Fuzzy >= 0.90        → direct match,    return role immediately
  Fuzzy >= 0.60        → suggest,         return top candidates
  Semantic >= 0.80     → direct match
  Semantic 0.60–0.80   → suggest
  < 0.60               → no match
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import get_settings
from src.models.role import RoleCategory

log = structlog.get_logger(__name__)
settings = get_settings()

_DIRECT_THRESHOLD        = 0.80
_SUGGEST_THRESHOLD       = 0.60
_FUZZY_DIRECT_THRESHOLD  = 0.90   # near-exact string match → return directly
_FUZZY_SUGGEST_THRESHOLD = 0.60   # plausible string match → surface as suggestions
_TOP_K_SUGGESTIONS       = 5


@dataclass
class RoleMatch:
    role_id: Optional[int]
    canonical_name: Optional[str]
    display_name: Optional[str]
    confidence: float
    match_type: str  # "exact_alias" | "fuzzy_match" | "semantic_direct" | "semantic_suggest" | "no_match"
    suggestions: list[dict]  # populated when match_type in ("semantic_suggest", "no_match")


@lru_cache(maxsize=1)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model)


def normalize_role(role_text: str, db: Session) -> RoleMatch:
    """
    Normalize a free-text role name to a canonical RoleCategory.

    Stage order is intentional:
      1. Exact match  — deterministic, zero false positives, no model load
      2. Fuzzy string — catches display-name variants and partial matches cheaply
      3. Embedding    — semantic fallback, only loads the model when truly needed
    """
    normalized = role_text.strip().lower()

    # ── Stage 1: exact alias / display_name match ────────────────────────────
    role = _alias_lookup(normalized, db)
    if role:
        log.debug(
            "role_normalize_exact_hit",
            input=role_text,
            role_id=role.role_id,
            display_name=role.display_name,
        )
        log.info("role_normalized", input=role_text, method="exact_alias", role_id=role.role_id)
        return RoleMatch(
            role_id=role.role_id,
            canonical_name=role.canonical_name,
            display_name=role.display_name,
            confidence=1.0,
            match_type="exact_alias",
            suggestions=[],
        )

    # ── Stage 2: fuzzy string match ──────────────────────────────────────────
    fuzzy_candidates = _fuzzy_search(normalized, db, top_k=_TOP_K_SUGGESTIONS)
    if fuzzy_candidates:
        best       = fuzzy_candidates[0]
        best_score = float(best["similarity"])

        if best_score >= _FUZZY_DIRECT_THRESHOLD:
            log.debug(
                "role_normalize_fuzzy_direct",
                input=role_text,
                matched=best["display_name"],
                score=round(best_score, 3),
            )
            log.info(
                "role_normalized",
                input=role_text,
                method="fuzzy_direct",
                role_id=best["role_id"],
                score=round(best_score, 3),
            )
            return RoleMatch(
                role_id=best["role_id"],
                canonical_name=best["canonical_name"],
                display_name=best["display_name"],
                confidence=best_score,
                match_type="fuzzy_match",
                suggestions=[],
            )

        if best_score >= _FUZZY_SUGGEST_THRESHOLD:
            suggestions = [
                {
                    "role_id":      c["role_id"],
                    "display_name": c["display_name"],
                    "similarity":   round(float(c["similarity"]), 3),
                }
                for c in fuzzy_candidates
            ]
            log.debug(
                "role_normalize_fuzzy_suggest",
                input=role_text,
                top_match=best["display_name"],
                score=round(best_score, 3),
                n_suggestions=len(suggestions),
            )
            log.info(
                "role_normalized",
                input=role_text,
                method="fuzzy_suggest",
                top_match=best["display_name"],
                score=round(best_score, 3),
            )
            return RoleMatch(
                role_id=None,
                canonical_name=None,
                display_name=None,
                confidence=best_score,
                match_type="semantic_suggest",
                suggestions=suggestions,
            )

    # ── Stage 3: embedding similarity (last resort — loads model) ────────────
    log.debug("role_normalize_embedding_fallback", input=role_text)
    model = _get_embedding_model()
    query_embedding = model.encode([role_text], normalize_embeddings=True)[0]
    candidates = _pgvector_search(query_embedding, db, top_k=_TOP_K_SUGGESTIONS)

    if not candidates:
        log.info("role_normalized", input=role_text, method="no_match")
        return RoleMatch(
            role_id=None,
            canonical_name=None,
            display_name=None,
            confidence=0.0,
            match_type="no_match",
            suggestions=[],
        )

    best       = candidates[0]
    similarity = float(best["similarity"])

    suggestions = [
        {
            "role_id":      c["role_id"],
            "display_name": c["display_name"],
            "similarity":   round(float(c["similarity"]), 3),
        }
        for c in candidates
    ]

    if similarity >= _DIRECT_THRESHOLD:
        log.debug(
            "role_normalize_semantic_direct",
            input=role_text,
            role_id=best["role_id"],
            similarity=round(similarity, 3),
        )
        log.info(
            "role_normalized",
            input=role_text,
            method="semantic_direct",
            role_id=best["role_id"],
            similarity=round(similarity, 3),
        )
        return RoleMatch(
            role_id=best["role_id"],
            canonical_name=best["canonical_name"],
            display_name=best["display_name"],
            confidence=similarity,
            match_type="semantic_direct",
            suggestions=[],
        )

    if similarity >= _SUGGEST_THRESHOLD:
        log.debug(
            "role_normalize_semantic_suggest",
            input=role_text,
            top_similarity=round(similarity, 3),
        )
        log.info(
            "role_normalized",
            input=role_text,
            method="semantic_suggest",
            top_similarity=round(similarity, 3),
        )
        return RoleMatch(
            role_id=None,
            canonical_name=None,
            display_name=None,
            confidence=similarity,
            match_type="semantic_suggest",
            suggestions=suggestions,
        )

    log.debug("role_normalize_no_match", input=role_text, top_similarity=round(similarity, 3))
    log.info("role_normalized", input=role_text, method="no_match", top_similarity=round(similarity, 3))
    return RoleMatch(
        role_id=None,
        canonical_name=None,
        display_name=None,
        confidence=similarity,
        match_type="no_match",
        suggestions=suggestions,
    )


def _alias_lookup(normalized_text: str, db: Session) -> Optional[RoleCategory]:
    """
    Exact match against canonical_name, display_name, or any alias — all case-insensitive.

    canonical_name is snake_case (e.g. 'backend_engineer'), so the ORM check only hits
    when the user types snake_case directly.  display_name and aliases cover all normal inputs.
    """
    # Fast path: snake_case canonical_name (e.g. programmatic callers)
    role = db.query(RoleCategory).filter(RoleCategory.canonical_name == normalized_text).first()
    if role:
        return role

    # Case-insensitive match against canonical_name, display_name, and all aliases
    row = db.execute(
        text(
            """
            SELECT role_id FROM role_categories
            WHERE lower(:name) = lower(canonical_name)
               OR lower(:name) = lower(display_name)
               OR lower(:name) = ANY(
                   SELECT lower(a) FROM unnest(coalesce(aliases, '{}')) AS a
               )
            LIMIT 1
            """
        ),
        {"name": normalized_text},
    ).first()

    if row:
        return db.query(RoleCategory).filter(RoleCategory.role_id == row.role_id).first()
    return None


def _fuzzy_search(normalized_text: str, db: Session, top_k: int = 5) -> list[dict]:
    """
    difflib string similarity against all role display_names and aliases.

    Scores every role and returns the top_k sorted by best ratio.  Runs before
    embedding search because it is cheaper (no model load, no vector index) and
    handles partial names, display-name variants, and minor typos correctly.
    """
    rows = db.execute(
        text(
            "SELECT role_id, canonical_name, display_name, aliases FROM role_categories"
        )
    ).fetchall()

    scored: list[tuple[float, dict]] = []
    for r in rows:
        best_ratio = difflib.SequenceMatcher(
            None, normalized_text, (r.display_name or "").lower()
        ).ratio()

        for alias in (r.aliases or []):
            ratio = difflib.SequenceMatcher(None, normalized_text, alias.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio

        if best_ratio > 0:
            scored.append((best_ratio, {
                "role_id":      r.role_id,
                "canonical_name": r.canonical_name,
                "display_name": r.display_name,
                "similarity":   best_ratio,
            }))

    scored.sort(key=lambda x: -x[0])
    return [entry for _, entry in scored[:top_k]]


def _pgvector_search(embedding: np.ndarray, db: Session, top_k: int = 5) -> list[dict]:
    vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding.tolist()) + "]"
    rows = db.execute(
        text(
            """
            SELECT role_id, canonical_name, display_name,
                   1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM role_categories
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
            """
        ),
        {"vec": vec_str, "k": top_k},
    ).fetchall()
    return [
        {
            "role_id":      r.role_id,
            "canonical_name": r.canonical_name,
            "display_name": r.display_name,
            "similarity":   r.similarity,
        }
        for r in rows
    ]
