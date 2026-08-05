"""
JD semantic concept expansion.

Job descriptions use natural-language phrases ("scalable ML systems",
"cross-functional collaboration") that don't appear as skill keywords but
imply specific technical requirements.

This module maps concept phrases to concrete skill names so the gap analysis
can surface requirements that raw keyword extraction would miss.

Zero DB queries, zero LLM calls. Expansion results are added to jd_profile
with reduced importance_score (0.4) so they don't dominate the gap score —
they are implied requirements, not explicit ones.
"""

from __future__ import annotations

# phrase (lowercase, substring match) → list of implied skill names
_JD_CONCEPT_MAP: dict[str, list[str]] = {
    # AI / ML
    "scalable ml": ["MLOps", "Docker", "Kubernetes", "System Design"],
    "production ml": ["MLOps", "Docker", "CI/CD", "Model Monitoring"],
    "ml pipeline": ["MLOps", "Apache Airflow", "Docker", "Python"],
    "multimodal ai": ["Computer Vision", "HuggingFace Transformers", "PyTorch"],
    "llm engineering": ["LangChain", "LlamaIndex", "RAG", "Prompt Engineering"],
    "generative ai": ["LangChain", "OpenAI API", "Prompt Engineering", "RAG"],
    "responsible ai": ["Model Monitoring", "Explainability", "Ethics"],
    "nlp pipeline": ["HuggingFace Transformers", "spaCy", "NLTK"],
    "computer vision": ["OpenCV", "PyTorch", "TensorFlow"],
    "model deployment": ["MLOps", "FastAPI", "Docker", "AWS"],
    "a/b testing": ["Statistical Analysis", "Python", "Experiment Design"],
    "recommendation system": ["Collaborative Filtering", "PyTorch", "Vector Search"],

    # Backend / Infrastructure
    "distributed system": ["Kafka", "Redis", "Kubernetes", "System Design"],
    "microservice": ["Docker", "Kubernetes", "REST API", "Message Queue"],
    "event-driven": ["Kafka", "RabbitMQ", "Celery", "Redis"],
    "data pipeline": ["Apache Airflow", "Kafka", "dbt", "Python"],
    "real-time processing": ["Kafka", "Redis", "WebSockets", "Spark"],
    "cloud native": ["AWS", "Kubernetes", "Docker", "Terraform"],
    "cloud infrastructure": ["AWS", "Terraform", "Kubernetes"],
    "high availability": ["Load Balancing", "Redis", "PostgreSQL", "Kubernetes"],
    "full stack": ["React", "FastAPI", "PostgreSQL", "REST API"],
    "api design": ["REST API", "GraphQL", "FastAPI", "OpenAPI"],

    # Soft / Process
    "fast-paced environment": ["Agile", "Scrum", "Adaptability"],
    "cross-functional": ["Collaboration", "Stakeholder Management", "Communication"],
    "communication skill": ["Communication", "Technical Writing"],
    "ownership mentality": ["Ownership", "Leadership", "Accountability"],
    "data-driven decision": ["Statistical Analysis", "SQL", "Data Visualization"],
    "agile methodology": ["Agile", "Scrum", "Jira"],

    # Security / Compliance
    "security best practice": ["OWASP", "Encryption", "IAM", "Penetration Testing"],
    "gdpr": ["Data Privacy", "Encryption", "Compliance"],
}


def expand_jd_concepts(jd_text: str) -> list[str]:
    """
    Return inferred skill names from natural-language concept phrases in JD text.

    Performs case-insensitive substring matching against _JD_CONCEPT_MAP.
    Deduplicates results so the same skill is not returned multiple times.
    Returns [] when no phrase matches — caller leaves jd_profile unchanged.
    """
    jd_lower = jd_text.lower()
    inferred: set[str] = set()
    for phrase, skills in _JD_CONCEPT_MAP.items():
        if phrase in jd_lower:
            inferred.update(skills)
    return list(inferred)
