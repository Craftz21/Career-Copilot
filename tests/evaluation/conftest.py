"""
Evaluation suite — shared fixtures and skip guards.

Two marker tiers:
  @pytest.mark.requires_db   — test needs a reachable PostgreSQL instance
  @pytest.mark.requires_seed — test needs seeded role_categories + skills data

Set TEST_DATABASE_URL to your seeded dev database before running:
    export TEST_DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname
    make audit
"""

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "requires_db: needs reachable PostgreSQL")
    config.addinivalue_line("markers", "requires_seed: needs seeded role + skill data")


# ---------------------------------------------------------------------------
# Marker enforcement — auto-skip rather than error
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    db_ok   = _db_reachable()
    seed_ok = _has_seed_data() if db_ok else False

    skip_db   = pytest.mark.skip(reason="TEST_DATABASE_URL not reachable — set env var and run `make seed`")
    skip_seed = pytest.mark.skip(reason="Database has no seed data — run `make seed` against TEST_DATABASE_URL")

    for item in items:
        if "requires_db" in item.keywords and not db_ok:
            item.add_marker(skip_db)
        elif "requires_seed" in item.keywords and not seed_ok:
            item.add_marker(skip_seed)


# ---------------------------------------------------------------------------
# Connectivity helpers (called once at collection time)
# ---------------------------------------------------------------------------

def _db_url() -> str:
    return os.getenv(
        "TEST_DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/career_copilot_test",
    )


def _db_reachable() -> bool:
    try:
        engine = create_engine(_db_url(), connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


def _has_seed_data() -> bool:
    try:
        engine = create_engine(_db_url(), connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            roles  = conn.execute(text("SELECT COUNT(*) FROM role_categories")).scalar() or 0
            skills = conn.execute(text("SELECT COUNT(*) FROM skills")).scalar() or 0
        engine.dispose()
        return roles > 0 and skills > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Session-scoped engine (reused across all evaluation tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def eval_engine():
    engine = create_engine(_db_url(), connect_args={"connect_timeout": 5})
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# Function-scoped transactional session — rolls back after every test
# ---------------------------------------------------------------------------

@pytest.fixture()
def eval_session(eval_engine):
    connection = eval_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


# ---------------------------------------------------------------------------
# Session-scoped read-only session (for fixtures that just read data)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def eval_ro(eval_engine):
    Session = sessionmaker(bind=eval_engine)
    session = Session()
    yield session
    session.close()
