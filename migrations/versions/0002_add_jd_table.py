"""Add jd_analyses table; make roadmaps.role_id nullable for JD mode

Revision ID: 0002_add_jd_table
Revises: 0001_initial_schema
Create Date: 2026-06-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0002_add_jd_table"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jd_analyses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), nullable=False),
        sa.Column("jd_raw_text", sa.Text(), nullable=False),
        sa.Column("company_name", sa.String(200), nullable=True),
        sa.Column("job_title", sa.String(200), nullable=True),
        sa.Column("jd_skills", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.session_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("session_id", name="uq_jd_analyses_session_id"),
    )
    op.create_index("idx_jd_analyses_session_id", "jd_analyses", ["session_id"])

    # JD-mode roadmaps have no canonical role_id — make the column nullable.
    op.alter_column("roadmaps", "role_id", nullable=True)


def downgrade() -> None:
    op.alter_column("roadmaps", "role_id", nullable=False)
    op.drop_index("idx_jd_analyses_session_id", table_name="jd_analyses")
    op.drop_table("jd_analyses")
