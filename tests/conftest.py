"""
Pytest fixtures shared across all tests.
Uses a real PostgreSQL test database (same schema, isolated data).
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Point at a test DB — override with TEST_DATABASE_URL env var
TEST_DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/career_copilot_test",
)

os.environ.setdefault("DATABASE_URL", TEST_DB_URL)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("GROQ_API_KEY", "test-key")


@pytest.fixture(scope="session")
def test_engine():
    from src.database import Base
    engine = create_engine(TEST_DB_URL)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def db_session(test_engine):
    """Provide a transactional DB session that rolls back after each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def client(test_engine, db_session):
    from src.database import get_db_session
    from src.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
