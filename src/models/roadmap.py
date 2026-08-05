import datetime
import uuid
from typing import Optional

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, SmallInteger, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class Roadmap(Base):
    __tablename__ = "roadmaps"

    roadmap_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    role_id: Mapped[Optional[int]] = mapped_column(
        SmallInteger, ForeignKey("role_categories.role_id"), nullable=True
    )
    duration: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False, default="v1")
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[dict] = mapped_column(JSONB, nullable=False)
    generation_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow
    )

    session: Mapped["UserSession"] = relationship("UserSession", back_populates="roadmap")
