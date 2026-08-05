import datetime
import uuid
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class UserSkill(Base):
    __tablename__ = "user_skills"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    skill_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("skills.skill_id", ondelete="CASCADE"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, default="embedding"
        # "embedding" | "alias_match" | "manual"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow
    )

    session: Mapped["UserSession"] = relationship("UserSession", back_populates="user_skills")
    skill: Mapped["Skill"] = relationship("Skill", back_populates="user_skills")

    __table_args__ = (
        # One row per session-skill pair
        {"schema": None},
    )
