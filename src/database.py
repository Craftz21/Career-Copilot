"""
Single SQLAlchemy engine and session factory for the entire application.
Both the API and Celery workers import from here — one engine, one pool.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings


def _build_engine():
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        pool_size=3,        # API + worker together stay within Neon free-tier limit (10)
        max_overflow=3,
        pool_timeout=30,    # fail fast rather than hang when pool is exhausted
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=not settings.is_production,
    )

    # Enable pgvector extension on first connection
    @event.listens_for(engine, "connect")
    def _enable_pgvector(dbapi_conn, _):
        with dbapi_conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            dbapi_conn.commit()

    return engine


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for database sessions. Handles commit/rollback automatically."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a DB session per request."""
    with get_db() as session:
        yield session


def check_db_connection() -> bool:
    """Health check: verify the database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
