"""
Shared test helpers for the evaluation suite.

All helpers are dependency-free except where noted.
DB helpers accept a SQLAlchemy Session and are pure functions over it.
"""

from __future__ import annotations

import io
import uuid
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# PDF / DOCX builders
# ---------------------------------------------------------------------------

def make_pdf_bytes(text_content: str) -> bytes:
    """
    Build a minimal single-column PDF from a plain-text string.
    Skips the test if PyMuPDF is not installed.
    """
    try:
        import fitz
    except ImportError:
        import pytest
        pytest.skip("PyMuPDF not installed — cannot build PDF fixture")

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 60
    for line in text_content.split("\n"):
        if y > 750:
            page = doc.new_page(width=612, height=792)
            y = 60
        page.insert_text((50, y), line.strip(), fontsize=10)
        y += 14
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def make_image_only_pdf() -> bytes:
    """
    PDF whose content is a rasterised bitmap — no selectable text.
    parse_resume() must raise ParseError on this (no OCR).
    """
    try:
        import fitz
    except ImportError:
        import pytest
        pytest.skip("PyMuPDF not installed")

    tmp = fitz.open()
    tmp_page = tmp.new_page(width=612, height=792)
    tmp_page.insert_text((50, 100), "SKILLS Python Docker PostgreSQL", fontsize=10)
    pix = tmp_page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    tmp.close()

    doc = fitz.open()
    pg = doc.new_page(width=612, height=792)
    pg.insert_image(fitz.Rect(0, 0, 612, 792), pixmap=pix)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def make_docx_bytes(text_content: str) -> bytes:
    """Build a minimal DOCX from plain text."""
    try:
        from docx import Document
    except ImportError:
        import pytest
        pytest.skip("python-docx not installed — cannot build DOCX fixture")

    doc = Document()
    for line in text_content.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Resume text fixtures — each represents a realistic persona
# ---------------------------------------------------------------------------

RESUME_MY = """
SKILLS
Python, FastAPI, REST API, PostgreSQL, Docker, ETL Pipelines, PyTorch, TensorFlow,
Machine Learning, Deep Learning, Git, Linux, Redis, SQL, Async Programming

EXPERIENCE
ML Platform Engineer — Samsung R&D | 2023-Present
Developed a PyTorch RNN simulator for digital twin forecasting, reducing error by 40%.
Built async FastAPI microservices handling 50k req/day with PostgreSQL and Redis.
Collaborated with Samsung R&D mentors on ETL pipeline improvements.
Deployed Docker containers to production; wrote CI/CD pipelines with GitHub Actions.

PROJECTS
CareerCopilot — FastAPI, PostgreSQL, Celery, pgvector. Open-source resume analyzer.
ETL Framework — Python ETL framework processing 1M records/day.

EDUCATION
B.Tech Computer Science | VIT | 2024
"""

RESUME_BEGINNER = """
EDUCATION
B.S. Computer Science | State University | 2024 (expected)

SKILLS
Python, Java, HTML, CSS, SQL (basic)

PROJECTS
Todo App — Python + Flask, SQLite database.
Calculator — Java command-line application.
"""

RESUME_FRONTEND = """
SKILLS
JavaScript, TypeScript, React, Next.js, Vue.js, CSS, HTML, Tailwind CSS,
Webpack, Vite, Jest, Cypress, Figma, Git

EXPERIENCE
Frontend Developer — WebAgency | 2021-Present
Built responsive web apps with React and TypeScript.
Integrated REST APIs and wrote Jest unit tests.
Led migration from Webpack to Vite, cutting build time by 60%.

PROJECTS
Portfolio site — Next.js, Tailwind CSS, deployed on Vercel.
"""

RESUME_BACKEND = """
SKILLS
Python, FastAPI, PostgreSQL, Docker, Redis, REST API, SQL, Linux,
Git, GitHub Actions, Celery, SQLAlchemy, Nginx

EXPERIENCE
Backend Engineer — TechCorp | 2021-Present
Designed REST APIs serving 100k users using FastAPI and PostgreSQL.
Implemented Redis caching layer reducing DB load by 70%.
Containerized services with Docker; automated deploys via GitHub Actions.
"""

RESUME_ML = """
SKILLS
Python, PyTorch, TensorFlow, scikit-learn, Hugging Face Transformers,
Machine Learning, Deep Learning, Pandas, NumPy, SQL, Docker, Git, MLflow

EXPERIENCE
ML Engineer — AIStartup | 2022-Present
Fine-tuned BERT models on proprietary NLP datasets; improved F1 by 12%.
Built training pipelines with PyTorch and MLflow experiment tracking.
Deployed models via FastAPI inference endpoints on Docker.

EDUCATION
M.S. Machine Learning | CMU | 2022
"""

RESUME_DATA_SCIENTIST = """
SKILLS
Python, R, SQL, Pandas, NumPy, scikit-learn, Tableau, Matplotlib,
Machine Learning, Statistical Analysis, A/B Testing, Jupyter Notebooks

EXPERIENCE
Data Scientist — DataCorp | 2022-Present
Analyzed large datasets using Python and Pandas; built Tableau dashboards.
Applied scikit-learn for classification models with 88% accuracy.
Ran A/B tests for product features; presented findings to leadership.
"""

RESUME_EMPTY = ""

RESUME_SKILLS_ONLY = """
SKILLS
Python FastAPI PostgreSQL Docker Redis Git Linux Machine Learning PyTorch TensorFlow
SQLAlchemy Celery Redis RabbitMQ Kubernetes GitHub Actions CI/CD REST API SQL
"""

RESUME_TWO_COLUMN_TEXT = """
SKILLS                          EXPERIENCE
Python                          Backend Engineer | TechCo | 2021-Present
Docker                          Built REST APIs with FastAPI
PostgreSQL                      Managed PostgreSQL databases
Redis                           Used Docker for containerization

EDUCATION
B.S. Computer Science | MIT | 2021
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_role_id(display_name: str, db: Session) -> Optional[int]:
    """Look up role_id by display_name, case-insensitive."""
    row = db.execute(
        text("SELECT role_id FROM role_categories WHERE LOWER(display_name) = LOWER(:n)"),
        {"n": display_name},
    ).first()
    return row.role_id if row else None


def get_skill_ids(skill_names: list[str], db: Session) -> dict[str, int]:
    """Return {display_name: skill_id} for names found in DB (case-insensitive)."""
    if not skill_names:
        return {}
    rows = db.execute(
        text(
            "SELECT skill_id, display_name FROM skills "
            "WHERE LOWER(display_name) = ANY(:names)"
        ),
        {"names": [n.lower() for n in skill_names]},
    ).fetchall()
    return {r.display_name: r.skill_id for r in rows}


def create_test_session(db: Session, role_id: int, target_role: str = "Test Role") -> str:
    """Insert a minimal UserSession row. Returns session_id string."""
    from src.models.session import UserSession
    sid = uuid.uuid4()
    obj = UserSession(
        session_id=sid,
        target_role=target_role,
        role_id=role_id,
        status="complete",
    )
    db.add(obj)
    db.flush()
    return str(sid)


def seed_user_skills(
    session_id: str,
    skill_ids: list[int],
    db: Session,
    confidence: float = 0.95,
) -> None:
    """Insert user_skills rows for a test session."""
    from src.models.user_skill import UserSkill
    for sid in skill_ids:
        db.add(UserSkill(
            session_id=uuid.UUID(session_id),
            skill_id=sid,
            confidence=confidence,
            source="alias_scan",
        ))
    db.flush()
