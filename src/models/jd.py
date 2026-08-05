import datetime
import uuid
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class JDAnalysis(Base):
    """
    Stores the raw job description text and JD-extracted skill profile for a session.
    Only exists for sessions created via POST /v1/jd/analyze (JD match mode).
    The presence of this record is the signal that a session is in JD mode.
    """

    __tablename__ = "jd_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    jd_raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    company_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    job_title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # [{skill_id, importance_score, display_name, category}] — persisted after extraction
    jd_skills: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.datetime.utcnow
    )

    session: Mapped["UserSession"] = relationship("UserSession", backref="jd_analysis")
