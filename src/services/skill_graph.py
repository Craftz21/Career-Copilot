"""
Semantic skill relationship graph.

Defines two relationship types between skills:
  ADJACENCY    — same paradigm, different implementation (e.g. PyTorch ↔ TensorFlow).
                 Expertise in one makes the other learnable in 1–3 weeks.
  TRANSFERABLE — higher-order skill implies foundation toward a lower-level one
                 (e.g. Algorithms → Data Structures, Deep Learning → Neural Networks).
                 Shortens the ramp-up from weeks to days.

Keys and values are normalised lowercase display names matching the skills DB.
Both directions of adjacency are defined explicitly so lookups are O(1).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Adjacency: bidirectional skill pairs
# ---------------------------------------------------------------------------
_ADJACENCY: dict[str, list[str]] = {
    # ML / DL frameworks
    "pytorch":      ["tensorflow", "keras", "jax"],
    "tensorflow":   ["pytorch", "keras", "jax"],
    "keras":        ["tensorflow", "pytorch"],
    "jax":          ["pytorch", "tensorflow"],

    # LLM / NLP tooling
    "hugging face transformers": ["transformers architecture", "natural language processing", "large language models"],
    "transformers architecture": ["hugging face transformers", "natural language processing", "computer vision"],
    "large language models":     ["hugging face transformers", "transformers architecture", "natural language processing"],
    "langchain":    ["llamaindex", "retrieval-augmented generation"],
    "llamaindex":   ["langchain", "retrieval-augmented generation"],
    "retrieval-augmented generation": ["embeddings", "langchain", "llamaindex"],

    # ML domains — share architecture, differ in data modality
    "natural language processing": ["computer vision", "speech recognition"],
    "computer vision":             ["natural language processing"],
    "reinforcement learning":      ["deep learning"],

    # Web — same paradigm, different ecosystem
    "react":        ["vue.js", "angular", "svelte", "next.js"],
    "vue.js":       ["react", "angular", "svelte", "nuxt.js"],
    "angular":      ["react", "vue.js"],
    "svelte":       ["react", "vue.js"],
    "next.js":      ["nuxt.js", "react"],
    "nuxt.js":      ["next.js", "vue.js"],

    # Backend frameworks — same REST/MVC pattern
    # REST API / API Development signals adjacency to any framework (protocol → implementation)
    # Microservices: anyone with backend API experience can learn microservice patterns quickly
    "rest api":         ["fastapi", "flask", "django", "express.js", "spring boot", "microservices"],
    "api development":  ["fastapi", "flask", "django", "express.js", "spring boot", "microservices"],
    "graphql":          ["rest api", "fastapi", "express.js"],
    "django":           ["fastapi", "flask", "express.js", "spring boot", "rest api", "api development", "microservices"],
    "fastapi":          ["flask", "django", "express.js", "rest api", "api development", "microservices"],
    "flask":            ["fastapi", "django", "rest api", "api development"],
    "express.js":       ["fastapi", "django", "flask", "rest api", "api development", "microservices"],
    "spring boot":      ["django", "express.js", "rest api", "api development", "microservices"],
    "microservices":    ["fastapi", "flask", "django", "express.js", "spring boot", "rest api", "api development"],

    # Cloud platforms — equivalent capability sets
    "amazon web services":    ["google cloud platform", "microsoft azure"],
    "google cloud platform":  ["amazon web services", "microsoft azure"],
    "microsoft azure":        ["amazon web services", "google cloud platform"],

    # Relational databases — any SQL knowledge implies adjacent DB proficiency
    "sql":          ["postgresql", "mysql", "sqlite"],
    "postgresql":   ["mysql", "sqlite", "sql"],
    "mysql":        ["postgresql", "sqlite", "sql"],
    "sqlite":       ["postgresql", "mysql", "sql"],

    # NoSQL
    "mongodb":          ["apache cassandra", "dynamodb"],
    "apache cassandra": ["mongodb", "dynamodb"],
    "dynamodb":         ["mongodb", "apache cassandra"],

    # Message brokers
    "apache kafka": ["rabbitmq"],
    "rabbitmq":     ["apache kafka"],

    # Orchestration / containers
    "kubernetes":    ["docker swarm", "docker"],
    "docker swarm":  ["kubernetes", "docker"],
    "docker":        ["docker compose", "kubernetes", "docker swarm"],
    "docker compose": ["docker"],

    # IaC
    "terraform":    ["ansible", "pulumi"],
    "ansible":      ["terraform", "pulumi"],
    "pulumi":       ["terraform", "ansible"],

    # CI/CD tools — same automation paradigm
    "github actions": ["gitlab ci", "jenkins", "circle ci"],
    "gitlab ci":      ["github actions", "jenkins"],
    "jenkins":        ["github actions", "gitlab ci"],
    "circle ci":      ["github actions"],

    # Languages — close siblings (same paradigm)
    # Python ↔ Java/Go: all are OOP/imperative languages; knowing one accelerates learning another
    "typescript":   ["javascript"],
    "javascript":   ["typescript"],
    "kotlin":       ["java", "scala"],
    "java":         ["kotlin", "scala", "c#", "python"],
    "c#":           ["java", "kotlin"],
    "scala":        ["java", "kotlin"],
    "rust":         ["c++", "go"],
    "go":           ["rust", "python"],
    "python":       ["java", "go"],
    "c++":          ["c", "rust"],
    "c":            ["c++"],

    # Testing
    "pytest":       ["jest"],
    "jest":         ["pytest", "vitest"],
    "vitest":       ["jest"],

    # Data processing
    "apache spark": ["dbt"],
    "dbt":          ["apache spark"],
}

# ---------------------------------------------------------------------------
# Transferability: unidirectional — higher-order implies foundation for lower
# ---------------------------------------------------------------------------
_TRANSFERABLE: dict[str, list[str]] = {
    # CS fundamentals
    "algorithms":                  ["data structures"],
    "data structures":             ["algorithms"],
    "compiler design":             ["algorithms", "data structures"],
    "operating systems concepts":  ["linux"],
    "computer networks":           ["networking", "http", "rest api"],

    # ML abstraction layers
    "deep learning":               ["natural language processing", "computer vision",
                                    "reinforcement learning"],
    "machine learning":            ["deep learning", "statistics"],
    "transformers architecture":   ["natural language processing", "computer vision"],
    "natural language processing": ["embeddings"],
    "computer vision":             ["embeddings"],

    # Platform / ops
    "mlops":              ["docker", "kubernetes", "github actions"],
    "devops":             ["docker", "github actions", "linux"],
    "system design":      ["distributed systems", "apache kafka", "redis"],
    "distributed systems": ["apache kafka", "redis"],

    # High-level → specific tools
    "retrieval-augmented generation": ["embeddings", "postgresql"],  # pgvector path
    "open source contribution":        ["git"],

    # Data roles
    "apache spark":       ["dbt"],

    # Backend / API transferability
    # Knowing REST API patterns implies quick ramp-up on any backend framework
    "rest api":           ["fastapi", "flask", "django", "express.js", "spring boot", "microservices"],
    "api development":    ["fastapi", "flask", "django", "express.js", "spring boot", "microservices"],
    "backend development": ["rest api", "api development", "microservices"],
    "microservices":      ["docker", "kubernetes"],  # microservices implies containerisation knowledge

    # SQL → DB design understanding
    "sql":                ["database design"],
}

# ---------------------------------------------------------------------------
# Pre-built reverse indexes for O(1) lookup
# ---------------------------------------------------------------------------
_ADJACENCY_REVERSE: dict[str, list[str]] = {}
for _src, _targets in _ADJACENCY.items():
    for _t in _targets:
        _ADJACENCY_REVERSE.setdefault(_t, []).append(_src)

_TRANSFERABLE_REVERSE: dict[str, list[str]] = {}
for _src, _targets in _TRANSFERABLE.items():
    for _t in _targets:
        _TRANSFERABLE_REVERSE.setdefault(_t, []).append(_src)


# ---------------------------------------------------------------------------
# Gap type classification
# ---------------------------------------------------------------------------
# Foundational: CS/ML theory that takes months to build, not weeks.
# Domain: specialisation areas — structured but learnable with a focused project.
# Tooling: specific libraries, platforms, or frameworks — learnable by doing.
# The distinction drives severity labelling and shortest-path time estimates.

_FOUNDATIONAL_SKILLS: frozenset[str] = frozenset({
    # CS core
    "algorithms", "data structures", "compiler design",
    "operating systems concepts", "computer networks", "discrete mathematics",
    "linear algebra", "probability and statistics", "calculus",
    "mathematics for machine learning", "object-oriented programming",
    "functional programming", "database design",
    # ML/AI theory
    "machine learning", "deep learning", "neural networks",
    "transformers architecture", "statistical learning",
    "information theory",
    # Systems
    "system design", "distributed systems", "computer architecture",
})

_DOMAIN_SKILLS: frozenset[str] = frozenset({
    # ML specialisations
    "natural language processing", "computer vision", "reinforcement learning",
    "speech recognition", "time series analysis", "anomaly detection",
    "recommendation systems", "information retrieval",
    # Ops/infra disciplines
    "mlops", "devops", "site reliability engineering",
    "cloud architecture", "software architecture", "microservices",
    "data engineering", "data science",
    # Security domains
    "application security", "cryptography", "network security",
    # Other
    "robotics", "bioinformatics", "quantum computing",
})


def classify_gap_type(display_name: str) -> str:
    """
    Return 'foundational' | 'domain' | 'tooling' for a missing skill.

    Foundational gaps are the most severe — they underpin everything else
    and cannot be closed with a weekend project. Domain gaps require structured
    study but are learnable with focused effort. Tooling gaps are the quickest
    to close and the least alarming to a recruiter.
    """
    norm = _normalize(display_name)
    if norm in _FOUNDATIONAL_SKILLS:
        return "foundational"
    if norm in _DOMAIN_SKILLS:
        return "domain"
    return "tooling"


def _normalize(name: str) -> str:
    name = name.strip().lower()
    # Strip trailing version suffixes: "MySQL 8.0" → "mysql", "Node.js 18" → "node.js"
    # This lets "MySQL 8.0" hit the "mysql" adjacency entry without a separate alias.
    name = re.sub(r"\s+v?\d[\d.x]*$", "", name)
    return name.strip()


def build_user_skill_index(display_names: list[str]) -> dict[str, str]:
    """Return {normalized_name: original_display_name} for O(1) graph lookups."""
    return {_normalize(name): name for name in display_names}


def classify_via_graph(
    missing_skill_name: str,
    user_skill_index: dict[str, str],
) -> tuple[str | None, str | None]:
    """
    Classify a missing skill using the semantic relationship graph.

    Args:
        missing_skill_name: display name of the skill that is missing
        user_skill_index:   {normalized_name: original_display_name}
                            for the candidate's matched skills

    Returns:
        (gap_status, via_skill_display_name)
        gap_status is "adjacent_expertise" | "transferable" | None
        None means no graph relationship found — caller falls through to DB hierarchy.

    Priority: adjacent_expertise > transferable.
    """
    missing_norm = _normalize(missing_skill_name)

    # Adjacency: check both forward (_ADJACENCY[missing]) and reverse indexes
    adj_candidates: list[str] = list(_ADJACENCY.get(missing_norm, []))
    adj_candidates.extend(_ADJACENCY_REVERSE.get(missing_norm, []))
    for adj in adj_candidates:
        original = user_skill_index.get(adj)
        if original is not None:
            return "adjacent_expertise", original

    # Transferable: a higher-order user skill implies this missing one
    for implier in _TRANSFERABLE_REVERSE.get(missing_norm, []):
        original = user_skill_index.get(implier)
        if original is not None:
            return "transferable", original

    return None, None
