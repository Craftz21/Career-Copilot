"""Initial schema with pgvector extension and all tables

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from pgvector.sqlalchemy import Vector

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "skill_categories",
        sa.Column("category_id", sa.SmallInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("parent_id", sa.SmallInteger(), nullable=True),
        sa.PrimaryKeyConstraint("category_id"),
        sa.UniqueConstraint("name"),
        sa.ForeignKeyConstraint(["parent_id"], ["skill_categories.category_id"]),
    )

    op.create_table(
        "skills",
        sa.Column("skill_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("canonical_name", sa.String(150), nullable=False),
        sa.Column("display_name", sa.String(150), nullable=False),
        sa.Column("category_id", sa.SmallInteger(), nullable=False),
        sa.Column("aliases", ARRAY(sa.Text()), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("skill_id"),
        sa.UniqueConstraint("canonical_name"),
        sa.ForeignKeyConstraint(["category_id"], ["skill_categories.category_id"]),
    )

    op.create_table(
        "role_categories",
        sa.Column("role_id", sa.SmallInteger(), autoincrement=True, nullable=False),
        sa.Column("canonical_name", sa.String(150), nullable=False),
        sa.Column("display_name", sa.String(150), nullable=False),
        sa.Column("domain", sa.String(100), nullable=True),
        sa.Column("aliases", ARRAY(sa.String(150)), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.PrimaryKeyConstraint("role_id"),
        sa.UniqueConstraint("canonical_name"),
    )

    op.create_table(
        "role_skill_profiles",
        sa.Column("role_id", sa.SmallInteger(), nullable=False),
        sa.Column("skill_id", sa.Integer(), nullable=False),
        sa.Column("job_count", sa.Integer(), nullable=False),
        sa.Column("total_jobs", sa.Integer(), nullable=False),
        sa.Column("frequency", sa.Numeric(5, 4), nullable=False),
        sa.Column("importance_score", sa.Numeric(5, 4), nullable=False),
        sa.Column("last_computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("role_id", "skill_id"),
        sa.ForeignKeyConstraint(["role_id"], ["role_categories.role_id"]),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.skill_id"]),
    )

    op.create_table(
        "companies",
        sa.Column("company_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("industry", sa.String(150), nullable=True),
        sa.PrimaryKeyConstraint("company_id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "jobs",
        sa.Column("job_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("role_id", sa.SmallInteger(), nullable=True),
        sa.Column("location", sa.String(255), nullable=True),
        sa.Column("is_remote", sa.Boolean(), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("posted_date", sa.Date(), nullable=True),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
        sa.ForeignKeyConstraint(["company_id"], ["companies.company_id"]),
        sa.ForeignKeyConstraint(["role_id"], ["role_categories.role_id"]),
    )

    op.create_table(
        "job_skills",
        sa.Column("job_id", sa.BigInteger(), nullable=False),
        sa.Column("skill_id", sa.Integer(), nullable=False),
        sa.Column("weight", sa.Numeric(5, 4), nullable=False),
        sa.Column("extraction_method", sa.String(50), nullable=False),
        sa.PrimaryKeyConstraint("job_id", "skill_id"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.job_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.skill_id"], ondelete="CASCADE"),
    )

    op.create_table(
        "learning_resources",
        sa.Column("resource_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("skill_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("platform", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=False),
        sa.Column("difficulty", sa.String(50), nullable=False),
        sa.Column("estimated_hours", sa.SmallInteger(), nullable=True),
        sa.PrimaryKeyConstraint("resource_id"),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.skill_id"], ondelete="CASCADE"),
    )

    op.create_table(
        "sessions",
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("target_role", sa.String(200), nullable=False),
        sa.Column("role_id", sa.SmallInteger(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("readiness_score", sa.SmallInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("session_id"),
        sa.ForeignKeyConstraint(["role_id"], ["role_categories.role_id"]),
    )

    op.create_table(
        "resumes",
        sa.Column("resume_id", UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("layout_type", sa.String(50), nullable=True),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("parsed_text", sa.Text(), nullable=True),
        sa.Column("sections", JSONB(), nullable=True),
        sa.Column("parse_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("resume_id"),
        sa.UniqueConstraint("session_id"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
    )

    op.create_table(
        "roadmaps",
        sa.Column("roadmap_id", UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("role_id", sa.SmallInteger(), nullable=False),
        sa.Column("duration", sa.String(50), nullable=False),
        sa.Column("prompt_version", sa.String(20), nullable=False),
        sa.Column("model_used", sa.String(100), nullable=False),
        sa.Column("content", JSONB(), nullable=False),
        sa.Column("generation_ms", sa.Integer(), nullable=True),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("roadmap_id"),
        sa.UniqueConstraint("session_id"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["role_categories.role_id"]),
    )

    op.create_table(
        "tasks",
        sa.Column("task_id", UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("task_type", sa.String(100), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("progress_pct", sa.SmallInteger(), nullable=False),
        sa.Column("progress_message", sa.String(255), nullable=True),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("task_id"),
        sa.UniqueConstraint("session_id"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
    )

    op.create_table(
        "user_skills",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("skill_id", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.skill_id"], ondelete="CASCADE"),
    )

    # Performance indexes
    op.create_index("idx_skills_is_active", "skills", ["is_active"])
    op.create_index("idx_sessions_expires_at", "sessions", ["expires_at"])
    op.create_index("idx_user_skills_session_id", "user_skills", ["session_id"])
    # O7: expression index on roadmap cache key — avoids full table scan on cache lookup
    op.execute("CREATE INDEX idx_roadmaps_cache_key ON roadmaps ((content->>'_cache_key'))")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_roadmaps_cache_key")
    op.drop_index("idx_user_skills_session_id", table_name="user_skills")
    op.drop_index("idx_sessions_expires_at", table_name="sessions")
    op.drop_index("idx_skills_is_active", table_name="skills")

    op.drop_table("user_skills")
    op.drop_table("tasks")
    op.drop_table("roadmaps")
    op.drop_table("resumes")
    op.drop_table("sessions")
    op.drop_table("learning_resources")
    op.drop_table("job_skills")
    op.drop_table("jobs")
    op.drop_table("companies")
    op.drop_table("role_skill_profiles")
    op.drop_table("role_categories")
    op.drop_table("skills")
    op.drop_table("skill_categories")
