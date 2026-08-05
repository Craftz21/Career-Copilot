"""
Alembic migration environment.

Uses the DATABASE_URL from application config so migrations run
against the same database as the app (including Neon in production).
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import all models so Alembic autogenerate can detect them
from src.database import Base  # noqa: F401
import src.models.skill  # noqa: F401
import src.models.role  # noqa: F401
import src.models.job  # noqa: F401
import src.models.session  # noqa: F401
import src.models.resume  # noqa: F401
import src.models.user_skill  # noqa: F401
import src.models.roadmap  # noqa: F401
import src.models.task  # noqa: F401
import src.models.resource  # noqa: F401

config = context.config

# Override sqlalchemy.url from env var if present (e.g. on Render)
database_url = os.getenv("DATABASE_URL")
if database_url:
    # Fix Render's "postgres://" prefix (SQLAlchemy requires "postgresql://")
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+psycopg2://", 1)
    config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
