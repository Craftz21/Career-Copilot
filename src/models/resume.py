import datetime
import uuid
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class Resume(Base):
    __tablename__ = "resumes"

    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    layout_type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
        # "single_column" | "multi_column" | "unknown"
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
        # pending | processing | complete | failed | scanned_pdf
    )
    parsed_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sections: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
        # { "experience": str, "education": str, "skills": str, "projects": str }
    )
    parse_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow
    )

    session: Mapped["UserSession"] = relationship("UserSession", back_populates="resume")
