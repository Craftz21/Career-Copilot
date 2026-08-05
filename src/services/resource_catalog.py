"""
Curated resource catalog for common skills.

Used to enrich LLM-generated and template-fallback roadmaps with real,
clickable learning resources. Zero DB or LLM dependencies.

Structure:
    RESOURCE_CATALOG  — primary skill → resource list mapping
    _SKILL_ALIASES    — lowercase display_name variants → canonical catalog key
    get_resources_for_skill(name) — alias-aware lookup, returns [] on miss
    enrich_roadmap_resources(roadmap) — post-process enrichment, in-place
"""

from __future__ import annotations

RESOURCE_CATALOG: dict[str, list[dict]] = {

    # ── AI / ML ───────────────────────────────────────────────────────────────

    "LangChain": [
        {
            "title": "LangChain Python — Official Tutorial",
            "platform": "LangChain",
            "url": "https://python.langchain.com/docs/get_started/introduction",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "LangChain Crash Course for Beginners",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=LbT1yp6quS8",
            "estimated_hours": 2,
            "type": "course",
            "difficulty": "Beginner",
        },
        {
            "title": "LangChain: Chat with Your Data",
            "platform": "DeepLearning.AI",
            "url": "https://learn.deeplearning.ai/langchain-chat-with-your-data",
            "estimated_hours": 3,
            "type": "course",
            "difficulty": "Intermediate",
        },
    ],

    "LlamaIndex": [
        {
            "title": "LlamaIndex — Getting Started",
            "platform": "LlamaIndex",
            "url": "https://docs.llamaindex.ai/en/stable/getting_started/starter_example/",
            "estimated_hours": 3,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "Building RAG Apps with LlamaIndex",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=TRjq7t2Ms5I",
            "estimated_hours": 2,
            "type": "course",
            "difficulty": "Intermediate",
        },
    ],

    "OpenAI API": [
        {
            "title": "OpenAI API Quickstart",
            "platform": "OpenAI",
            "url": "https://platform.openai.com/docs/quickstart",
            "estimated_hours": 2,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "OpenAI Cookbook — Practical Examples",
            "platform": "GitHub",
            "url": "https://github.com/openai/openai-cookbook",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Intermediate",
        },
        {
            "title": "Building Systems with the ChatGPT API",
            "platform": "DeepLearning.AI",
            "url": "https://learn.deeplearning.ai/building-systems-with-chatgpt",
            "estimated_hours": 3,
            "type": "course",
            "difficulty": "Beginner",
        },
    ],

    "HuggingFace Transformers": [
        {
            "title": "HuggingFace NLP Course",
            "platform": "HuggingFace",
            "url": "https://huggingface.co/learn/nlp-course/chapter1/1",
            "estimated_hours": 8,
            "type": "course",
            "difficulty": "Beginner",
        },
        {
            "title": "HuggingFace Transformers — Official Docs",
            "platform": "HuggingFace",
            "url": "https://huggingface.co/docs/transformers/index",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Intermediate",
        },
    ],

    "Vector Search": [
        {
            "title": "Vector Databases: from Embeddings to Applications",
            "platform": "DeepLearning.AI",
            "url": "https://learn.deeplearning.ai/vector-databases-embeddings-applications",
            "estimated_hours": 3,
            "type": "course",
            "difficulty": "Beginner",
        },
        {
            "title": "Pinecone Learning Center",
            "platform": "Pinecone",
            "url": "https://www.pinecone.io/learn/",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Beginner",
        },
    ],

    "RAG": [
        {
            "title": "Building RAG Agents with LLMs",
            "platform": "DeepLearning.AI",
            "url": "https://learn.deeplearning.ai/building-rag-agents-with-llms",
            "estimated_hours": 4,
            "type": "course",
            "difficulty": "Intermediate",
        },
        {
            "title": "Retrieval Augmented Generation — LangChain Docs",
            "platform": "LangChain",
            "url": "https://python.langchain.com/docs/use_cases/question_answering/",
            "estimated_hours": 2,
            "type": "documentation",
            "difficulty": "Intermediate",
        },
    ],

    "Prompt Engineering": [
        {
            "title": "ChatGPT Prompt Engineering for Developers",
            "platform": "DeepLearning.AI",
            "url": "https://learn.deeplearning.ai/chatgpt-prompt-eng",
            "estimated_hours": 3,
            "type": "course",
            "difficulty": "Beginner",
        },
        {
            "title": "Prompt Engineering Guide",
            "platform": "DAIR.AI",
            "url": "https://www.promptingguide.ai/",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Beginner",
        },
    ],

    # ── Backend ───────────────────────────────────────────────────────────────

    "FastAPI": [
        {
            "title": "FastAPI — Official Tutorial",
            "platform": "FastAPI",
            "url": "https://fastapi.tiangolo.com/tutorial/",
            "estimated_hours": 5,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "FastAPI Full Course for Beginners",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=7t2alSnE2-I",
            "estimated_hours": 3,
            "type": "course",
            "difficulty": "Beginner",
        },
    ],

    "PostgreSQL": [
        {
            "title": "PostgreSQL Tutorial",
            "platform": "PostgreSQLTutorial.com",
            "url": "https://www.postgresqltutorial.com/",
            "estimated_hours": 8,
            "type": "course",
            "difficulty": "Beginner",
        },
        {
            "title": "PostgreSQL Official Documentation",
            "platform": "PostgreSQL",
            "url": "https://www.postgresql.org/docs/current/tutorial.html",
            "estimated_hours": 6,
            "type": "documentation",
            "difficulty": "Beginner",
        },
    ],

    "Redis": [
        {
            "title": "Redis University — RU101: Introduction to Redis",
            "platform": "Redis University",
            "url": "https://university.redis.com/courses/ru101/",
            "estimated_hours": 6,
            "type": "course",
            "difficulty": "Beginner",
        },
        {
            "title": "Redis Documentation — Getting Started",
            "platform": "Redis",
            "url": "https://redis.io/docs/get-started/",
            "estimated_hours": 3,
            "type": "documentation",
            "difficulty": "Beginner",
        },
    ],

    "Celery": [
        {
            "title": "Celery — Getting Started Guide",
            "platform": "Celery",
            "url": "https://docs.celeryq.dev/en/stable/getting-started/introduction.html",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "Celery with FastAPI — Async Task Processing",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=7Dzf86LCbcI",
            "estimated_hours": 2,
            "type": "course",
            "difficulty": "Intermediate",
        },
    ],

    "Docker": [
        {
            "title": "Docker Official Get Started Guide",
            "platform": "Docker",
            "url": "https://docs.docker.com/get-started/",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "Docker Tutorial for Beginners — Full Course",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=pTFZFxd5hOI",
            "estimated_hours": 4,
            "type": "course",
            "difficulty": "Beginner",
        },
    ],

    # ── Cloud / DevOps ────────────────────────────────────────────────────────

    "AWS": [
        {
            "title": "AWS Cloud Practitioner Essentials",
            "platform": "AWS Training",
            "url": "https://aws.amazon.com/training/digital/aws-cloud-practitioner-essentials/",
            "estimated_hours": 6,
            "type": "course",
            "difficulty": "Beginner",
        },
        {
            "title": "AWS Getting Started Resource Center",
            "platform": "AWS",
            "url": "https://aws.amazon.com/getting-started/",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Beginner",
        },
    ],

    "Kubernetes": [
        {
            "title": "Kubernetes Basics — Official Interactive Tutorial",
            "platform": "Kubernetes",
            "url": "https://kubernetes.io/docs/tutorials/kubernetes-basics/",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "Kubernetes Full Course for Beginners",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=d6WC5n9G_sM",
            "estimated_hours": 4,
            "type": "course",
            "difficulty": "Beginner",
        },
    ],

    # ── Testing ───────────────────────────────────────────────────────────────

    "Playwright": [
        {
            "title": "Playwright for Python — Official Docs",
            "platform": "Playwright",
            "url": "https://playwright.dev/python/docs/intro",
            "estimated_hours": 3,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "Playwright Tutorial for Beginners",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=wawbt1cATsk",
            "estimated_hours": 2,
            "type": "course",
            "difficulty": "Beginner",
        },
    ],

    "Selenium": [
        {
            "title": "Selenium with Python — Official Documentation",
            "platform": "Selenium",
            "url": "https://selenium-python.readthedocs.io/",
            "estimated_hours": 5,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "Selenium Python Full Course",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=Xjv1sY630Uc",
            "estimated_hours": 3,
            "type": "course",
            "difficulty": "Beginner",
        },
    ],

    "pytest": [
        {
            "title": "pytest — Official Getting Started Guide",
            "platform": "pytest",
            "url": "https://docs.pytest.org/en/stable/getting-started.html",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "Python Testing with pytest — TestDriven.io",
            "platform": "TestDriven.io",
            "url": "https://testdriven.io/blog/modern-tox-pytest-ci/",
            "estimated_hours": 3,
            "type": "course",
            "difficulty": "Intermediate",
        },
    ],

    "JUnit": [
        {
            "title": "JUnit 5 User Guide",
            "platform": "JUnit",
            "url": "https://junit.org/junit5/docs/current/user-guide/",
            "estimated_hours": 4,
            "type": "documentation",
            "difficulty": "Beginner",
        },
        {
            "title": "JUnit 5 Tutorial for Beginners",
            "platform": "YouTube",
            "url": "https://www.youtube.com/watch?v=flpmSXVTqBI",
            "estimated_hours": 3,
            "type": "course",
            "difficulty": "Beginner",
        },
    ],
}


# Lowercase display_name variants → canonical catalog key.
# Covers common DB storage variants and abbreviations.
_SKILL_ALIASES: dict[str, str] = {
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "llama index": "LlamaIndex",
    "llama-index": "LlamaIndex",
    "openai": "OpenAI API",
    "openai api": "OpenAI API",
    "open ai": "OpenAI API",
    "huggingface": "HuggingFace Transformers",
    "huggingface transformers": "HuggingFace Transformers",
    "hugging face": "HuggingFace Transformers",
    "transformers": "HuggingFace Transformers",
    "hf transformers": "HuggingFace Transformers",
    "vector search": "Vector Search",
    "vector database": "Vector Search",
    "vector databases": "Vector Search",
    "semantic search": "Vector Search",
    "pgvector": "Vector Search",
    "rag": "RAG",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "prompt engineering": "Prompt Engineering",
    "prompting": "Prompt Engineering",
    "fastapi": "FastAPI",
    "fast api": "FastAPI",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "psql": "PostgreSQL",
    "redis": "Redis",
    "celery": "Celery",
    "docker": "Docker",
    "aws": "AWS",
    "amazon web services": "AWS",
    "amazon aws": "AWS",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "playwright": "Playwright",
    "selenium": "Selenium",
    "pytest": "pytest",
    "junit": "JUnit",
    "junit5": "JUnit",
    "junit 5": "JUnit",
}


def get_resources_for_skill(skill_name: str) -> list[dict]:
    """
    Return curated resources for a skill name, normalizing via alias table.
    Returns [] when the skill has no catalog entry — caller uses LLM resources.
    """
    key = skill_name.strip()
    if key in RESOURCE_CATALOG:
        return RESOURCE_CATALOG[key]
    canonical = _SKILL_ALIASES.get(key.lower())
    if canonical:
        return RESOURCE_CATALOG.get(canonical, [])
    return []


def enrich_roadmap_resources(roadmap: dict) -> dict:
    """
    Post-process a generated roadmap dict to replace placeholder resources
    with curated catalog entries wherever a match exists.

    Handles both week-based (legacy) and phase-based (v2) roadmap shapes.
    Uses `or []` guards because Pydantic serializes Optional[list] fields as
    None (not absent), so `.get(key, [])` would return None for those keys.
    """
    # Week-based roadmaps (template fallback / v1 prompt)
    for week in (roadmap.get("weeks") or []):
        curated: list[dict] = []
        seen: set[str] = set()
        for skill_name in (week.get("skills") or []):
            for res in get_resources_for_skill(skill_name):
                if res["title"] not in seen:
                    seen.add(res["title"])
                    curated.append(res)
        if curated:
            week["resources"] = curated[:4]

    # Phase-based roadmaps (v2 prompt) — enrich per-phase resources
    for phase in (roadmap.get("phases") or []):
        curated = []
        seen = set()
        # phases have closes_gaps (skill names) instead of a dedicated skills list
        skill_names = (phase.get("closes_gaps") or [])
        for skill_name in skill_names:
            for res in get_resources_for_skill(skill_name):
                if res["title"] not in seen:
                    seen.add(res["title"])
                    curated.append(res)
        if curated:
            # Prepend catalog resources; keep LLM-generated ones as fallback
            existing = (phase.get("resources") or [])
            existing_titles = {r.get("title") for r in existing}
            merged = curated[:3] + [r for r in existing if r.get("title") not in existing_titles]
            phase["resources"] = merged[:5]

    return roadmap
