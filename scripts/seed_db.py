"""
Database seeding script.

Run once after `alembic upgrade head` to populate:
  1. skill_categories   — hierarchical skill taxonomy
  2. skills             — 500+ skills with embeddings
  3. role_categories    — canonical role definitions with embeddings
  4. role_skill_profiles — role→skill importance mappings
  5. learning_resources — curated resources for fallback roadmap

Usage:
  python scripts/seed_db.py
  python scripts/seed_db.py --reset   (drops and re-seeds everything)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from sqlalchemy import text

from src.config import get_settings
from src.database import Base, SessionLocal, engine

log = structlog.get_logger(__name__)
settings = get_settings()

DATA_DIR = Path(__file__).parent.parent / "data"
SKILLS_CSV = DATA_DIR / "skills_master.csv"
ROLES_CSV = DATA_DIR / "roles.csv"


def main(reset: bool = False) -> None:
    log.info("seed_start", reset=reset)

    if reset:
        log.warning("seed_reset_dropping_data")
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE role_skill_profiles, user_skills, learning_resources, roadmaps, skills, role_categories, skill_categories RESTART IDENTITY CASCADE"))

    with SessionLocal() as db:
        # 1. Seed skill categories
        category_map = _seed_skill_categories(db)

        # 2. Seed skills (with embeddings)
        skill_map = _seed_skills(db, category_map)

        # 3. Seed roles (with embeddings)
        role_map = _seed_roles(db)

        # 4. Seed role-skill profiles (hardcoded importance mappings)
        _seed_role_skill_profiles(db, role_map, skill_map)

        # 5. Seed learning resources
        _seed_learning_resources(db, skill_map)

        db.commit()

    log.info("seed_complete")


# ---------------------------------------------------------------------------
# Skill Categories
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("programming_languages", "Programming Languages", None),
    ("backend_frameworks", "Backend Frameworks", None),
    ("frontend_frameworks", "Frontend Frameworks & Libraries", None),
    ("frontend_technologies", "Frontend Technologies", None),
    ("mobile_development", "Mobile Development", None),
    ("databases_relational", "Relational Databases", "databases"),
    ("databases_nosql", "NoSQL Databases", "databases"),
    ("databases_vector", "Vector Databases", "databases"),
    ("databases", "Databases", None),
    ("cloud_platforms", "Cloud Platforms", None),
    ("cloud_aws", "AWS Services", "cloud_platforms"),
    ("cloud_gcp", "GCP Services", "cloud_platforms"),
    ("cloud_azure", "Azure Services", "cloud_platforms"),
    ("containers_orchestration", "Containers & Orchestration", None),
    ("cicd_tools", "CI/CD Tools", None),
    ("infrastructure_as_code", "Infrastructure as Code", None),
    ("monitoring_observability", "Monitoring & Observability", None),
    ("message_queues", "Message Queues & Streaming", None),
    ("ai_ml_frameworks", "AI/ML Frameworks", None),
    ("ai_ml_concepts", "AI/ML Concepts", None),
    ("data_engineering", "Data Engineering", None),
    ("data_science", "Data Science", None),
    ("mlops", "MLOps", None),
    ("testing", "Testing", None),
    ("system_design", "System Design", None),
    ("methodologies", "Methodologies & Practices", None),
    ("version_control", "Version Control", None),
    ("security", "Security", None),
    ("web_servers", "Web Servers & Proxies", None),
    ("build_tools", "Build Tools", None),
    ("api_standards", "API Standards", None),
    ("operating_systems", "Operating Systems", None),
    ("soft_skills", "Soft Skills", None),
    ("computer_science", "Computer Science Fundamentals", None),
]


def _seed_skill_categories(db) -> dict[str, int]:
    """Insert skill categories. Returns {canonical_name: category_id}."""
    from src.models.skill import SkillCategory

    category_map: dict[str, int] = {}
    existing = {c.name: c.category_id for c in db.query(SkillCategory).all()}

    # First pass: insert top-level categories
    for canonical, display, parent_canonical in CATEGORIES:
        if display in existing:
            category_map[canonical] = existing[display]
            continue
        cat = SkillCategory(name=display)
        db.add(cat)
        db.flush()
        category_map[canonical] = cat.category_id

    # Second pass: set parent_id
    for canonical, display, parent_canonical in CATEGORIES:
        if parent_canonical and parent_canonical in category_map:
            cat = db.query(SkillCategory).filter(SkillCategory.name == display).first()
            if cat:
                cat.parent_id = category_map[parent_canonical]

    db.flush()
    log.info("skill_categories_seeded", count=len(category_map))
    return category_map


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def _seed_skills(db, category_map: dict[str, int]) -> dict[str, int]:
    """Seed skills from CSV with embeddings. Returns {canonical_name: skill_id}."""
    from sentence_transformers import SentenceTransformer
    from src.models.skill import Skill

    model = SentenceTransformer(settings.embedding_model)

    skill_map: dict[str, int] = {}
    existing = {s.canonical_name: s.skill_id for s in db.query(Skill).all()}

    rows = []
    with open(SKILLS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Batch embed all display names + aliases for better representation
    texts_to_embed = [
        f"{r['display_name']} {r['category']} {r['aliases'].replace('|', ' ')}"
        for r in rows
    ]
    log.info("embedding_skills", count=len(texts_to_embed))
    embeddings = model.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=True, batch_size=64)

    new_count = 0
    for i, row in enumerate(rows):
        canonical = row["canonical_name"].strip()
        if canonical in existing:
            skill_map[canonical] = existing[canonical]
            continue

        cat_key = row["category"].strip()
        category_id = category_map.get(cat_key)
        if category_id is None:
            # Create missing category on the fly so seed never crashes
            from src.models.skill import SkillCategory
            new_cat = SkillCategory(name=cat_key.replace("_", " ").title())
            db.add(new_cat)
            db.flush()
            category_map[cat_key] = new_cat.category_id
            category_id = new_cat.category_id
        aliases = [a.strip() for a in row["aliases"].split("|") if a.strip()]
        embedding_list = embeddings[i].tolist()

        skill = Skill(
            canonical_name=canonical,
            display_name=row["display_name"].strip(),
            category_id=category_id,
            aliases=aliases,
            embedding=embedding_list,
            is_active=True,
        )
        db.add(skill)
        db.flush()
        skill_map[canonical] = skill.skill_id
        new_count += 1

    log.info("skills_seeded", new=new_count, total=len(skill_map))
    return skill_map


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

def _seed_roles(db) -> dict[str, int]:
    """Seed role categories with embeddings. Returns {canonical_name: role_id}."""
    from sentence_transformers import SentenceTransformer
    from src.models.role import RoleCategory

    model = SentenceTransformer(settings.embedding_model)
    role_map: dict[str, int] = {}
    existing = {r.canonical_name: r.role_id for r in db.query(RoleCategory).all()}

    rows = []
    with open(ROLES_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    texts = [
        f"{r['display_name']} {r['domain']} {r['aliases'].replace('|', ' ')}"
        for r in rows
    ]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    updated_count = 0
    new_count = 0
    for i, row in enumerate(rows):
        canonical = row["canonical_name"].strip()
        aliases = [a.strip() for a in row["aliases"].split("|") if a.strip()]
        if canonical in existing:
            role_map[canonical] = existing[canonical]
            # Sync aliases from CSV without regenerating embeddings.
            # Allows roles.csv updates (new aliases, display_name tweaks) to
            # propagate on a normal re-seed without --reset.
            role_obj = db.query(RoleCategory).filter(
                RoleCategory.canonical_name == canonical
            ).first()
            if role_obj and set(role_obj.aliases or []) != set(aliases):
                role_obj.aliases = aliases
                updated_count += 1
            continue

        role = RoleCategory(
            canonical_name=canonical,
            display_name=row["display_name"].strip(),
            domain=row["domain"].strip(),
            aliases=aliases,
            embedding=embeddings[i].tolist(),
        )
        db.add(role)
        db.flush()
        role_map[canonical] = role.role_id
        new_count += 1

    log.info("roles_seeded", new=new_count, aliases_updated=updated_count, total=len(role_map))
    return role_map


# ---------------------------------------------------------------------------
# Role-Skill Profiles
# ---------------------------------------------------------------------------

# Curated skill importance mappings per role.
# importance_score: 0.0–1.0 (1.0 = must-have, 0.5 = nice-to-have)
# frequency: how often this skill appears in job postings for this role (0–1)
ROLE_SKILL_PROFILES = {
    "backend_engineer": [
        ("python", 0.95, 0.80), ("fastapi", 0.80, 0.65), ("flask", 0.70, 0.55),
        ("django", 0.75, 0.60), ("expressjs", 0.75, 0.65), ("springboot", 0.80, 0.70),
        ("go", 0.70, 0.55), ("postgresql", 0.90, 0.85), ("redis", 0.85, 0.80),
        ("docker", 0.90, 0.90), ("rest_api", 0.95, 0.95), ("sql", 0.90, 0.90),
        ("git", 0.90, 0.95), ("microservices", 0.80, 0.75), ("caching", 0.75, 0.70),
        ("api_design", 0.85, 0.80), ("pytest", 0.80, 0.75), ("tdd", 0.70, 0.65),
        ("celery", 0.65, 0.55), ("grpc", 0.60, 0.50), ("kubernetes", 0.70, 0.65),
        ("aws", 0.75, 0.70), ("jwt", 0.75, 0.70), ("openapi", 0.70, 0.65),
        ("linux", 0.80, 0.80), ("mongodb", 0.70, 0.60), ("graphql", 0.65, 0.55),
    ],
    "frontend_engineer": [
        ("javascript", 0.98, 0.98), ("typescript", 0.90, 0.85), ("react", 0.90, 0.88),
        ("html", 0.98, 0.98), ("css", 0.98, 0.95), ("nextjs", 0.80, 0.75),
        ("tailwindcss", 0.75, 0.70), ("jest", 0.80, 0.75), ("webpack", 0.70, 0.65),
        ("vite", 0.70, 0.65), ("git", 0.90, 0.95), ("rest_api", 0.85, 0.85),
        ("graphql", 0.70, 0.65), ("vuejs", 0.75, 0.70), ("angular", 0.70, 0.65),
        ("css", 0.95, 0.95), ("playwright", 0.70, 0.65), ("storybook", 0.65, 0.60),
        ("pwa", 0.60, 0.55), ("webassembly", 0.50, 0.40),
    ],
    "fullstack_engineer": [
        ("javascript", 0.95, 0.95), ("typescript", 0.85, 0.80), ("react", 0.85, 0.85),
        ("nextjs", 0.80, 0.78), ("nodejs", 0.80, 0.80), ("expressjs", 0.75, 0.70),
        ("python", 0.75, 0.70), ("postgresql", 0.85, 0.82), ("mongodb", 0.70, 0.65),
        ("rest_api", 0.90, 0.90), ("docker", 0.80, 0.78), ("git", 0.92, 0.95),
        ("html", 0.95, 0.95), ("css", 0.90, 0.90), ("tailwindcss", 0.70, 0.65),
        ("aws", 0.70, 0.65), ("redis", 0.70, 0.65), ("sql", 0.80, 0.80),
    ],
    "data_scientist": [
        ("python", 0.98, 0.98), ("pandas", 0.95, 0.95), ("numpy", 0.92, 0.92),
        ("scikit_learn", 0.90, 0.90), ("machine_learning", 0.95, 0.95),
        ("sql", 0.85, 0.85), ("statistics", 0.88, 0.88), ("jupyter", 0.88, 0.88),
        ("tensorflow", 0.75, 0.70), ("pytorch", 0.75, 0.72), ("matplotlib", 0.85, 0.85),
        ("seaborn", 0.80, 0.78), ("deep_learning", 0.75, 0.70), ("nlp", 0.70, 0.65),
        ("feature_engineering", 0.88, 0.85), ("model_evaluation", 0.90, 0.88),
        ("git", 0.85, 0.88), ("r", 0.70, 0.60), ("tableau", 0.65, 0.60),
        ("plotly", 0.70, 0.65), ("time_series", 0.75, 0.70), ("anomaly_detection", 0.65, 0.60),
    ],
    "ml_engineer": [
        ("python", 0.98, 0.98), ("pytorch", 0.92, 0.90), ("tensorflow", 0.85, 0.82),
        ("scikit_learn", 0.88, 0.86), ("machine_learning", 0.98, 0.98),
        ("deep_learning", 0.90, 0.88), ("docker", 0.88, 0.85), ("kubernetes", 0.80, 0.75),
        ("mlflow", 0.80, 0.78), ("wandb", 0.75, 0.72), ("pandas", 0.88, 0.88),
        ("numpy", 0.88, 0.88), ("sql", 0.80, 0.78), ("git", 0.90, 0.92),
        ("ray", 0.72, 0.68), ("aws", 0.80, 0.78), ("embeddings", 0.80, 0.78),
        ("mlops_concepts", 0.85, 0.82), ("feature_engineering", 0.85, 0.82),
        ("model_evaluation", 0.90, 0.88), ("dvc", 0.70, 0.65), ("fastapi", 0.75, 0.72),
    ],
    "ai_engineer": [
        ("python", 0.98, 0.98), ("llm", 0.95, 0.95), ("prompt_engineering", 0.92, 0.92),
        ("langchain", 0.88, 0.88), ("openai_sdk", 0.88, 0.88), ("rag", 0.90, 0.88),
        ("embeddings", 0.90, 0.90), ("vector_search", 0.88, 0.86),
        ("huggingface", 0.85, 0.83), ("fastapi", 0.85, 0.82),
        ("docker", 0.85, 0.82), ("postgresql", 0.78, 0.75), ("pgvector", 0.80, 0.78),
        ("git", 0.90, 0.92), ("fine_tuning", 0.80, 0.78), ("agentic_ai", 0.80, 0.78),
        ("llamaindex", 0.80, 0.78), ("generative_ai", 0.92, 0.90),
        ("aws", 0.78, 0.75), ("redis", 0.75, 0.72), ("celery", 0.70, 0.65),
        ("pydantic", 0.85, 0.82),
    ],
    "devops_engineer": [
        ("docker", 0.98, 0.98), ("kubernetes", 0.95, 0.92), ("linux", 0.95, 0.95),
        ("bash", 0.92, 0.92), ("github_actions", 0.88, 0.88), ("terraform", 0.88, 0.85),
        ("aws", 0.88, 0.88), ("ansible", 0.80, 0.78), ("prometheus", 0.82, 0.80),
        ("grafana", 0.82, 0.80), ("git", 0.92, 0.95), ("jenkins", 0.78, 0.75),
        ("nginx", 0.80, 0.78), ("helm", 0.82, 0.80), ("python", 0.78, 0.75),
        ("gitlab_ci", 0.78, 0.75), ("opentelemetry", 0.72, 0.68), ("elk_stack", 0.72, 0.70),
        ("argocd", 0.78, 0.75), ("networking", 0.82, 0.80), ("aws_ecs", 0.78, 0.75),
    ],
    "data_engineer": [
        ("python", 0.95, 0.95), ("sql", 0.95, 0.95), ("apache_spark", 0.88, 0.85),
        ("apache_airflow", 0.85, 0.82), ("postgresql", 0.85, 0.82), ("dbt", 0.82, 0.80),
        ("aws", 0.82, 0.80), ("docker", 0.82, 0.80), ("git", 0.88, 0.90),
        ("apache_kafka", 0.80, 0.78), ("pandas", 0.88, 0.88), ("snowflake", 0.80, 0.78),
        ("databricks", 0.78, 0.75), ("data_pipeline", 0.90, 0.88), ("data_modeling", 0.85, 0.82),
        ("delta_lake", 0.72, 0.68), ("great_expectations", 0.70, 0.65),
        ("aws_glue", 0.72, 0.68), ("gcp_bigquery", 0.72, 0.70), ("linux", 0.80, 0.80),
    ],
    "cloud_engineer": [
        ("aws", 0.95, 0.95), ("terraform", 0.92, 0.90), ("kubernetes", 0.90, 0.88),
        ("docker", 0.90, 0.90), ("linux", 0.90, 0.90), ("python", 0.82, 0.80),
        ("bash", 0.85, 0.85), ("aws_ec2", 0.85, 0.85), ("aws_s3", 0.85, 0.85),
        ("aws_rds", 0.82, 0.80), ("aws_lambda", 0.82, 0.80), ("aws_iam", 0.85, 0.85),
        ("aws_cloudwatch", 0.80, 0.78), ("networking", 0.85, 0.85), ("git", 0.88, 0.90),
        ("prometheus", 0.78, 0.75), ("helm", 0.82, 0.80), ("ansible", 0.78, 0.75),
        ("ssl_tls", 0.80, 0.78), ("github_actions", 0.78, 0.75),
    ],
    "security_engineer": [
        ("owasp", 0.95, 0.95), ("penetration_testing", 0.85, 0.82), ("python", 0.82, 0.82),
        ("linux", 0.88, 0.88), ("networking", 0.88, 0.88), ("ssl_tls", 0.88, 0.88),
        ("oauth2", 0.85, 0.85), ("jwt", 0.82, 0.80), ("sast_dast", 0.82, 0.80),
        ("devsecops", 0.80, 0.78), ("secrets_management", 0.80, 0.78),
        ("container_security", 0.78, 0.75), ("api_security", 0.80, 0.78),
        ("zero_trust", 0.75, 0.72), ("compliance", 0.75, 0.72),
        ("docker", 0.78, 0.75), ("kubernetes", 0.75, 0.72), ("git", 0.85, 0.88),
    ],
}


def _seed_role_skill_profiles(db, role_map: dict[str, int], skill_map: dict[str, int]) -> None:
    from src.models.role import RoleSkillProfile

    count = 0
    seen_pairs: set[tuple[int, int]] = set()
    for role_canonical, skill_list in ROLE_SKILL_PROFILES.items():
        role_id = role_map.get(role_canonical)
        if not role_id:
            continue

        # Clear existing profile for this role
        db.query(RoleSkillProfile).filter(RoleSkillProfile.role_id == role_id).delete()

        for skill_canonical, importance, frequency in skill_list:
            skill_id = skill_map.get(skill_canonical)
            if not skill_id:
                log.warning("seed_skill_not_found", skill=skill_canonical)
                continue
            pair = (role_id, skill_id)
            if pair in seen_pairs:
                log.warning("seed_duplicate_profile_skipped", role_id=role_id, skill=skill_canonical)
                continue
            seen_pairs.add(pair)
            total_jobs = 100
            db.add(RoleSkillProfile(
                role_id=role_id,
                skill_id=skill_id,
                job_count=round(frequency * total_jobs),
                total_jobs=total_jobs,
                importance_score=importance,
                frequency=frequency,
            ))
            count += 1

    db.flush()
    log.info("role_skill_profiles_seeded", count=count)


# ---------------------------------------------------------------------------
# Learning Resources (fallback for LLM failures)
# ---------------------------------------------------------------------------

RESOURCES = [
    # Python
    ("python", "Python Official Tutorial", "python.org", "documentation", "beginner", 10),
    ("python", "Automate the Boring Stuff with Python", "automatetheboringstuff.com", "book", "beginner", 20),
    ("python", "Python for Everybody (Coursera)", "Coursera", "course", "beginner", 30),
    # FastAPI
    ("fastapi", "FastAPI Official Docs", "fastapi.tiangolo.com", "documentation", "intermediate", 8),
    ("fastapi", "FastAPI Full Course — freeCodeCamp", "YouTube", "course", "intermediate", 6),
    # Django
    ("django", "Django Official Tutorial", "djangoproject.com", "documentation", "beginner", 8),
    ("django", "Django for Everybody (Coursera)", "Coursera", "course", "beginner", 25),
    # React
    ("react", "React Official Docs (react.dev)", "react.dev", "documentation", "beginner", 10),
    ("react", "Full Stack Open (Part 1)", "fullstackopen.com", "course", "intermediate", 20),
    # SQL / PostgreSQL
    ("sql", "SQLZoo", "sqlzoo.net", "tutorial", "beginner", 8),
    ("postgresql", "PostgreSQL Official Docs", "postgresql.org", "documentation", "intermediate", 12),
    # Docker
    ("docker", "Docker Getting Started Guide", "docs.docker.com", "documentation", "beginner", 6),
    ("docker", "Docker & Kubernetes Full Course — TechWorld with Nana", "YouTube", "course", "intermediate", 10),
    # Kubernetes
    ("kubernetes", "Kubernetes Official Docs", "kubernetes.io", "documentation", "intermediate", 15),
    ("kubernetes", "CKA Study Guide — killer.sh", "killer.sh", "course", "advanced", 40),
    # Machine Learning
    ("machine_learning", "Machine Learning Specialization — Andrew Ng", "Coursera", "course", "beginner", 80),
    ("scikit_learn", "scikit-learn User Guide", "scikit-learn.org", "documentation", "intermediate", 10),
    # PyTorch
    ("pytorch", "PyTorch Official Tutorials", "pytorch.org", "documentation", "intermediate", 15),
    ("pytorch", "Deep Learning with PyTorch — freeCodeCamp", "YouTube", "course", "intermediate", 20),
    # LangChain
    ("langchain", "LangChain Official Docs", "python.langchain.com", "documentation", "intermediate", 10),
    ("langchain", "LangChain Crash Course — freeCodeCamp", "YouTube", "course", "intermediate", 4),
    # AWS
    ("aws", "AWS Cloud Practitioner Essentials", "AWS Skill Builder", "course", "beginner", 12),
    ("aws", "AWS Solutions Architect Associate — Adrian Cantrill", "learn.cantrill.io", "course", "intermediate", 80),
    # Terraform
    ("terraform", "HashiCorp Terraform Tutorials", "developer.hashicorp.com", "tutorial", "beginner", 8),
    # Git
    ("git", "Pro Git Book (free)", "git-scm.com", "book", "beginner", 15),
    ("git", "Learn Git Branching", "learngitbranching.js.org", "tutorial", "beginner", 4),
    # TypeScript
    ("typescript", "TypeScript Handbook", "typescriptlang.org", "documentation", "intermediate", 8),
    ("typescript", "Total TypeScript", "totaltypescript.com", "course", "intermediate", 15),
    # System Design
    ("system_design_interviews", "System Design Primer (GitHub)", "GitHub", "documentation", "intermediate", 20),
    ("system_design_interviews", "Designing Data-Intensive Applications", "O'Reilly", "book", "advanced", 40),
]


def _seed_learning_resources(db, skill_map: dict[str, int]) -> None:
    from src.models.resource import LearningResource

    count = 0
    for skill_canonical, title, platform, rtype, difficulty, hours in RESOURCES:
        skill_id = skill_map.get(skill_canonical)
        if not skill_id:
            continue
        existing = db.query(LearningResource).filter(
            LearningResource.skill_id == skill_id,
            LearningResource.title == title,
        ).first()
        if existing:
            continue
        db.add(LearningResource(
            skill_id=skill_id,
            title=title,
            platform=platform,
            resource_type=rtype,
            difficulty=difficulty,
            estimated_hours=hours,
        ))
        count += 1

    db.flush()
    log.info("learning_resources_seeded", count=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the CareerCopilot database.")
    parser.add_argument("--reset", action="store_true", help="Truncate all seed tables before seeding.")
    args = parser.parse_args()
    main(reset=args.reset)
