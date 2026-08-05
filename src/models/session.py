import datetime
import uuid
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class UserSession(Base):
    """
    Represents a user's interaction session — no authentication required.
    The session_id IS the user's identity. Results are available for 7 days.
    """

    __tablename__ = "sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    target_role: Mapped[str] = mapped_column(String(200), nullable=False)
    role_id: Mapped[Optional[int]] = mapped_column(
        SmallInteger, ForeignKey("role_categories.role_id"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
        # pending | parsing | generating | complete | failed
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    readiness_score: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)  # 0–100
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    resume: Mapped[Optional["Resume"]] = relationship(
        "Resume", back_populates="session", uselist=False
    )
    user_skills: Mapped[list["UserSkill"]] = relationship(
        "UserSkill", back_populates="session"
    )
    roadmap: Mapped[Optional["Roadmap"]] = relationship(
        "Roadmap", back_populates="session", uselist=False
    )
    task: Mapped[Optional["Task"]] = relationship(
        "Task", back_populates="session", uselist=False
    )

    def is_expired(self) -> bool:
        return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc) > self.expires_at
