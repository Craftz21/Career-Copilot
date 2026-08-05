import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, Integer, Numeric, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class Company(Base):
    __tablename__ = "companies"

    company_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    industry: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)

    jobs: Mapped[list["Job"]] = relationship("Job", back_populates="company")


class Job(Base):
    __tablename__ = "jobs"

    job_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("companies.company_id"), nullable=False
    )
    role_id: Mapped[Optional[int]] = mapped_column(
        SmallInteger, ForeignKey("role_categories.role_id"), nullable=True
    )
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_remote: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    posted_date: Mapped[Optional[datetime.date]] = mapped_column(Date, nullable=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False, default="seed")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow
    )

    company: Mapped[Company] = relationship("Company", back_populates="jobs")
    role: Mapped[Optional["RoleCategory"]] = relationship("RoleCategory", back_populates="jobs")
    job_skills: Mapped[list["JobSkill"]] = relationship("JobSkill", back_populates="job")


class JobSkill(Base):
    __tablename__ = "job_skills"

    job_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True
    )
    skill_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("skills.skill_id", ondelete="CASCADE"), primary_key=True
    )
    weight: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False, default=1.0)
    extraction_method: Mapped[str] = mapped_column(
        String(50), nullable=False, default="embedding"
    )

    job: Mapped[Job] = relationship("Job", back_populates="job_skills")
    skill: Mapped["Skill"] = relationship("Skill", back_populates="job_skills")
