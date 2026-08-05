import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class SkillCategory(Base):
    __tablename__ = "skill_categories"

    category_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    parent_id: Mapped[Optional[int]] = mapped_column(
        SmallInteger, ForeignKey("skill_categories.category_id"), nullable=True
    )

    skills: Mapped[list["Skill"]] = relationship("Skill", back_populates="category")


class Skill(Base):
    __tablename__ = "skills"

    skill_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(150), nullable=False)
    category_id: Mapped[int] = mapped_column(
        SmallInteger, ForeignKey("skill_categories.category_id"), nullable=False
    )
    aliases: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow
    )

    category: Mapped[SkillCategory] = relationship("SkillCategory", back_populates="skills")
    job_skills: Mapped[list["JobSkill"]] = relationship("JobSkill", back_populates="skill")
    user_skills: Mapped[list["UserSkill"]] = relationship("UserSkill", back_populates="skill")
    role_profiles: Mapped[list["RoleSkillProfile"]] = relationship(
        "RoleSkillProfile", back_populates="skill"
    )
    resources: Mapped[list["LearningResource"]] = relationship(
        "LearningResource", back_populates="skill"
    )

    def __repr__(self) -> str:
        return f"<Skill {self.canonical_name}>"
