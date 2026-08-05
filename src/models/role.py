import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, SmallInteger, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class RoleCategory(Base):
    __tablename__ = "role_categories"

    role_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(150), nullable=False)
    domain: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # e.g. "backend", "frontend", "ml", "data", "devops", "mobile", "fullstack"
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String(150)), nullable=False, default=list)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384), nullable=True)

    jobs: Mapped[list["Job"]] = relationship("Job", back_populates="role")
    skill_profiles: Mapped[list["RoleSkillProfile"]] = relationship(
        "RoleSkillProfile", back_populates="role"
    )


class RoleSkillProfile(Base):
    """
    Pre-computed: for each role, how important is each skill?
    This is the hot table for gap analysis — avoids live aggregation.
    Refreshed after each ETL run.
    """

    __tablename__ = "role_skill_profiles"

    role_id: Mapped[int] = mapped_column(
        SmallInteger, ForeignKey("role_categories.role_id"), primary_key=True
    )
    skill_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("skills.skill_id"), primary_key=True
    )
    job_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_jobs: Mapped[int] = mapped_column(Integer, nullable=False)
    frequency: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    # frequency = job_count / total_jobs
    importance_score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    # importance_score = frequency * avg_weight * recency_factor
    last_computed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow
    )

    role: Mapped[RoleCategory] = relationship("RoleCategory", back_populates="skill_profiles")
    skill: Mapped["Skill"] = relationship("Skill", back_populates="role_profiles")
