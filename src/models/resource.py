from typing import Optional

from sqlalchemy import ForeignKey, Integer, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class LearningResource(Base):
    """
    Curated learning resources for common skills.
    Used as a fallback when LLM generation fails,
    and as a quality check against LLM-suggested resources.
    """

    __tablename__ = "learning_resources"

    resource_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    skill_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("skills.skill_id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    platform: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(
        String(50), nullable=False
        # "course" | "book" | "documentation" | "tutorial" | "project"
    )
    difficulty: Mapped[str] = mapped_column(
        String(50), nullable=False
        # "beginner" | "intermediate" | "advanced"
    )
    estimated_hours: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    skill: Mapped["Skill"] = relationship("Skill", back_populates="resources")
