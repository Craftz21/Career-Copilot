import datetime
import uuid
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class Task(Base):
    """User-facing task status table. Separate from Celery's internal result backend."""

    __tablename__ = "tasks"

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    task_type: Mapped[str] = mapped_column(
        String(100), nullable=False
        # "parse_resume" | "generate_roadmap"
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="queued"
        # queued | processing | complete | failed
    )
    progress_pct: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    progress_message: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    result: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    queued_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow
    )
    started_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    session: Mapped["UserSession"] = relationship("UserSession", back_populates="task")
