"""
Two-pass skill extraction pipeline.

Pass 1 — Alias pre-scan (exact/fuzzy string match against all known aliases).
          Fast, deterministic, confidence = 0.95.

Pass 2 — Embedding ANN search via pgvector on 80-word overlapping text chunks.
          Catches paraphrased / informal skill mentions.

Score fusion:
  - Section weight multiplies the raw similarity score.
  - Skills appearing in the 'skills' section get 3× boost.
  - Deduplication: keep highest-scored mention per skill_id.
  - Threshold: final_score >= SKILL_EXTRACTION_THRESHOLD (default 0.38).

Performance optimisations applied:
  O1 — Alias map + compiled regex patterns cached at process level (built once
       from DB on first call, reused for all subsequent tasks in the same worker).
  O2 — pgvector ANN queries batched: all chunk embeddings sent in a single
       SQL round-trip via json_to_recordset + LATERAL, instead of one query
       per chunk.
"""

import json as _json
import re
import threading
from functools import lru_cache
from typing import Optional

import numpy as np
import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import get_settings
from src.models.skill import Skill
from src.services.resume_parser import get_section_weight

log = structlog.get_logger(__name__)
settings = get_settings()

_CHUNK_SIZE = 80      # words per chunk
_CHUNK_OVERLAP = 20   # word overlap between chunks
_TOP_K = 5            # ANN candidates per chunk

# ── O1: Process-level alias/pattern caches ────────────────────────────────────
# Populated lazily on the first extract_skills() call; thread-safe via a lock.
# Celery workers are processes, so this survives across all tasks in one slot.
_alias_map_cache: Optional[dict[str, int]] = None      # alias_lower → skill_id
_sorted_aliases_cache: Optional[list[str]] = None      # sorted longest-first
_pattern_cache: Optional[dict[str, re.Pattern]] = None # alias → compiled regex
_cache_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_skills(
    sections: dict[str, str],
    raw_text: str,
    db: Session,
) -> list[dict]:
    """
    Returns a list of dicts:
        [{"skill_id": int, "confidence": float, "source": str}, ...]
    Deduplicated — one entry per skill_id (highest confidence wins).
    """
    _prime_caches(db)   # O1: no-op on warm worker; one DB hit on cold start

    candidates: dict[int, dict] = {}  # skill_id → best match

    # --- Pass 1: alias scan per section (O1: uses pre-compiled patterns) ---
    for section_name, section_text in sections.items():
        weight = get_section_weight(section_name)
        matches = _alias_scan_cached(section_text)
        for skill_id, raw_conf in matches:
            score = min(raw_conf * weight, 1.0)
            _update_best(candidates, skill_id, score, "alias_match")

    # --- Pass 2: embedding ANN — batch all chunks in one pgvector round-trip ---
    model = _get_embedding_model()

    # Collect (section_name, chunk_text) pairs across all sections
    all_section_chunks: list[tuple[str, str]] = []
    for section_name, section_text in sections.items():
        for chunk in _make_chunks(section_text, _CHUNK_SIZE, _CHUNK_OVERLAP):
            all_section_chunks.append((section_name, chunk))

    if all_section_chunks:
        chunk_texts = [c for _, c in all_section_chunks]
        # Single model.encode() call for all chunks (already batched)
        all_embeddings = model.encode(
            chunk_texts, normalize_embeddings=True, show_progress_bar=False
        )
        # O2: one SQL round-trip for all embeddings
        batch_results = _pgvector_search_batch(list(all_embeddings), db, top_k=_TOP_K)

        for (section_name, _), ann_hits in zip(all_section_chunks, batch_results):
            weight = get_section_weight(section_name)
            for skill_id, similarity in ann_hits:
                score = min(float(similarity) * weight, 1.0)
                _update_best(candidates, skill_id, score, "embedding")

    threshold = settings.skill_extraction_threshold
    results = [
        {"skill_id": sid, "confidence": round(info["confidence"], 4), "source": info["source"]}
        for sid, info in candidates.items()
        if info["confidence"] >= threshold
    ]

    log.info(
        "skills_extracted",
        total_candidates=len(candidates),
        above_threshold=len(results),
        threshold=threshold,
    )
    return results


# ---------------------------------------------------------------------------
# O1: Cache management
# ---------------------------------------------------------------------------

def _prime_caches(db: Session) -> None:
    """
    Build alias map + compiled regex patterns on first call; no-op thereafter.
    Double-checked locking keeps it safe under Celery's threading model.
    """
    global _alias_map_cache, _sorted_aliases_cache, _pattern_cache
    if _alias_map_cache is not None:
        return
    with _cache_lock:
        if _alias_map_cache is not None:  # another thread may have built it
            return
        alias_map = _build_alias_map(db)
        sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
        patterns = {
            alias: re.compile(_make_boundary_pattern(alias))
            for alias in alias_map
            if alias
        }
        # Atomic triple-assign under GIL
        _alias_map_cache = alias_map
        _sorted_aliases_cache = sorted_aliases
        _pattern_cache = patterns
        log.info("alias_cache_primed", alias_count=len(alias_map))


def invalidate_alias_cache() -> None:
    """Force rebuild on next extract_skills() call. Call after re-seeding skills."""
    global _alias_map_cache, _sorted_aliases_cache, _pattern_cache
    with _cache_lock:
        _alias_map_cache = None
        _sorted_aliases_cache = None
        _pattern_cache = None


# ---------------------------------------------------------------------------
# Alias pre-scan
# ---------------------------------------------------------------------------

def _build_alias_map(db: Session) -> dict[str, int]:
    """
    Returns {alias_lower: skill_id} for all active skills.
    Includes canonical_name and display_name as aliases.
    """
    skills = db.query(Skill).filter(Skill.is_active == True).all()  # noqa: E712
    alias_map: dict[str, int] = {}
    for skill in skills:
        alias_map[skill.canonical_name.lower()] = skill.skill_id
        alias_map[skill.display_name.lower()] = skill.skill_id
        if skill.aliases:
            for alias in skill.aliases:
                alias_map[alias.lower().strip()] = skill.skill_id
    return alias_map


def _make_boundary_pattern(alias: str) -> str:
    r"""
    Build a word-boundary-aware regex for an alias.

    \b is the boundary between \w and \W. It silently fails when the alias
    starts or ends with a non-word char — e.g. c++, c#, f#, .net c# all have
    non-word chars at the boundary position, so \b never fires there.
    Use negative lookarounds in those cases instead.
    """
    escaped = re.escape(alias)
    prefix = r"\b" if alias[0].isalnum() or alias[0] == "_" else r"(?<!\w)"
    suffix = r"\b" if alias[-1].isalnum() or alias[-1] == "_" else r"(?!\w)"
    return prefix + escaped + suffix


def _alias_scan_cached(text: str) -> list[tuple[int, float]]:
    """
    Alias scan using the process-level pre-compiled pattern cache (O1).
    Must be called after _prime_caches().
    """
    text_lower = text.lower()
    found: dict[int, float] = {}
    for alias in _sorted_aliases_cache:  # type: ignore[union-attr]
        if _pattern_cache[alias].search(text_lower):  # type: ignore[index]
            skill_id = _alias_map_cache[alias]  # type: ignore[index]
            if skill_id not in found:
                found[skill_id] = 0.95
    return list(found.items())


def _alias_scan(text: str, alias_map: dict[str, int]) -> list[tuple[int, float]]:
    """
    Alias scan with an explicit alias map (used by evaluate_parser.py and tests).
    Compiles patterns on each call — not for the hot path.
    """
    text_lower = text.lower()
    found: dict[int, float] = {}
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if not alias:
            continue
        if re.search(_make_boundary_pattern(alias), text_lower):
            skill_id = alias_map[alias]
            if skill_id not in found:
                found[skill_id] = 0.95
    return list(found.items())


# ---------------------------------------------------------------------------
# O2: Batched pgvector search
# ---------------------------------------------------------------------------

def _pgvector_search_batch(
    embeddings: list[np.ndarray],
    db: Session,
    top_k: int = 5,
) -> list[list[tuple[int, float]]]:
    """
    Query pgvector for top-k neighbours for ALL embeddings in one SQL round-trip.

    Uses json_to_recordset to pass all query vectors as a single JSON parameter,
    then a LATERAL join to run the ANN search per vector inside the same query.
    Reduces N network round-trips to 1 regardless of chunk count.

    Returns a list of length len(embeddings), each element being a list of
    (skill_id, similarity) tuples sorted by similarity descending.
    """
    if not embeddings:
        return []

    records = [
        {
            "idx": i,
            "vec": "[" + ",".join(f"{v:.6f}" for v in emb.tolist()) + "]",
        }
        for i, emb in enumerate(embeddings)
    ]

    rows = db.execute(
        text(
            """
            SELECT q.idx::int AS idx, s.skill_id, s.similarity
            FROM json_to_recordset(:records) AS q(idx int, vec text)
            CROSS JOIN LATERAL (
                SELECT skill_id,
                       1 - (embedding <=> CAST(q.vec AS vector)) AS similarity
                FROM   skills
                WHERE  is_active = true AND embedding IS NOT NULL
                ORDER  BY embedding <=> CAST(q.vec AS vector)
                LIMIT  :k
            ) AS s
            ORDER BY q.idx, s.similarity DESC
            """
        ),
        {"records": _json.dumps(records), "k": top_k},
    ).fetchall()

    results: list[list[tuple[int, float]]] = [[] for _ in embeddings]
    for row in rows:
        results[row.idx].append((row.skill_id, float(row.similarity)))
    return results


def _pgvector_search(
    embedding: np.ndarray,
    db: Session,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    Single-embedding pgvector search. Kept for evaluate_parser.py and tests.
    The hot path (extract_skills) uses _pgvector_search_batch instead.
    """
    vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding.tolist()) + "]"
    rows = db.execute(
        text(
            """
            SELECT skill_id, 1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM skills
            WHERE is_active = true AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
            """
        ),
        {"vec": vec_str, "k": top_k},
    ).fetchall()
    return [(row.skill_id, row.similarity) for row in rows]


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _make_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_best(
    candidates: dict[int, dict],
    skill_id: int,
    score: float,
    source: str,
) -> None:
    existing = candidates.get(skill_id)
    if existing is None or score > existing["confidence"]:
        candidates[skill_id] = {"confidence": score, "source": source}
