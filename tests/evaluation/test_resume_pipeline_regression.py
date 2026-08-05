"""
Phase 2 — Resume Pipeline Regression Suite

Tests parse_resume() across 10 resume classes.
No DB required for parsing; skill extraction tests are DB-marked.

Verified per fixture:
  - parser returns without exception
  - sections dict is present and non-empty
  - raw_text is non-empty
  - layout_type matches expected value
  - known keywords appear in extracted text
"""

import io
import pytest

from src.services.resume_parser import ParseError, parse_resume
from .helpers import (
    RESUME_BACKEND,
    RESUME_BEGINNER,
    RESUME_DATA_SCIENTIST,
    RESUME_EMPTY,
    RESUME_FRONTEND,
    RESUME_ML,
    RESUME_MY,
    RESUME_SKILLS_ONLY,
    RESUME_TWO_COLUMN_TEXT,
    make_docx_bytes,
    make_pdf_bytes,
    make_image_only_pdf,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parsed(text: str) -> dict:
    """Parse a synthetic PDF and return the result dict."""
    return parse_resume(make_pdf_bytes(text), "resume.pdf")


# ---------------------------------------------------------------------------
# Phase 2-A: Parser shape contract
# ---------------------------------------------------------------------------

class TestParserOutputShape:
    """Every successful parse must return the same dict shape."""

    def test_my_resume_shape(self):
        result = _parsed(RESUME_MY)
        assert "raw_text"    in result
        assert "sections"    in result
        assert "layout_type" in result
        assert "char_count"  in result
        assert result["layout_type"] == "pdf"

    def test_char_count_matches_raw_text(self):
        result = _parsed(RESUME_BACKEND)
        assert result["char_count"] == len(result["raw_text"])

    def test_sections_is_dict(self):
        result = _parsed(RESUME_ML)
        assert isinstance(result["sections"], dict)

    def test_sections_not_empty_for_structured_resume(self):
        result = _parsed(RESUME_MY)
        assert len(result["sections"]) >= 1, "Expected at least one section from structured resume"


# ---------------------------------------------------------------------------
# Phase 2-B: Content extraction per persona
# ---------------------------------------------------------------------------

class TestContentExtraction:
    def test_my_resume_contains_fastapi(self):
        result = _parsed(RESUME_MY)
        assert "fastapi" in result["raw_text"].lower() or "FastAPI" in result["raw_text"]

    def test_my_resume_contains_pytorch(self):
        result = _parsed(RESUME_MY)
        assert "pytorch" in result["raw_text"].lower()

    def test_beginner_resume_parses(self):
        result = _parsed(RESUME_BEGINNER)
        assert len(result["raw_text"]) > 0

    def test_frontend_resume_contains_react(self):
        result = _parsed(RESUME_FRONTEND)
        assert "react" in result["raw_text"].lower()

    def test_backend_resume_contains_postgresql(self):
        result = _parsed(RESUME_BACKEND)
        assert "postgresql" in result["raw_text"].lower() or "postgres" in result["raw_text"].lower()

    def test_ml_resume_contains_pytorch(self):
        result = _parsed(RESUME_ML)
        assert "pytorch" in result["raw_text"].lower()

    def test_data_scientist_resume_contains_sql(self):
        result = _parsed(RESUME_DATA_SCIENTIST)
        assert "sql" in result["raw_text"].lower()

    def test_two_column_resume_parses_without_crash(self):
        """Two-column layout may interleave text, but must not crash."""
        result = _parsed(RESUME_TWO_COLUMN_TEXT)
        assert len(result["raw_text"]) > 0

    def test_skills_only_resume_parses(self):
        result = _parsed(RESUME_SKILLS_ONLY)
        assert len(result["raw_text"]) > 0


# ---------------------------------------------------------------------------
# Phase 2-C: Empty / degenerate inputs
# ---------------------------------------------------------------------------

class TestDegenerateInputs:
    def test_empty_pdf_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_resume(b"", "resume.pdf")

    def test_invalid_bytes_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_resume(b"NOT A PDF FILE", "resume.pdf")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ParseError, match="Unsupported"):
            parse_resume(b"bytes", "resume.xlsx")

    def test_image_only_pdf_raises_parse_error(self):
        """PDF with no selectable text must raise ParseError (no OCR)."""
        pdf_bytes = make_image_only_pdf()
        with pytest.raises(ParseError):
            parse_resume(pdf_bytes, "resume.pdf")

    def test_empty_resume_text_produces_minimal_output(self):
        """
        A PDF with only whitespace/blank lines must not crash.
        It may raise ParseError (no content) — either is acceptable.
        """
        try:
            result = _parsed(RESUME_EMPTY)
            # If it doesn't raise, ensure at least the shape is there
            assert "raw_text" in result
        except ParseError:
            pass  # also acceptable


# ---------------------------------------------------------------------------
# Phase 2-D: DOCX parsing
# ---------------------------------------------------------------------------

class TestDocxParsing:
    def test_docx_parses_without_crash(self):
        docx_bytes = make_docx_bytes(RESUME_BACKEND)
        result = parse_resume(docx_bytes, "resume.docx")
        assert "raw_text" in result
        assert len(result["raw_text"]) > 0

    def test_docx_layout_type_is_docx(self):
        docx_bytes = make_docx_bytes(RESUME_ML)
        result = parse_resume(docx_bytes, "resume.docx")
        assert result["layout_type"] == "docx"

    def test_docx_contains_expected_keywords(self):
        docx_bytes = make_docx_bytes(RESUME_ML)
        result = parse_resume(docx_bytes, "resume.docx")
        assert "pytorch" in result["raw_text"].lower()


# ---------------------------------------------------------------------------
# Phase 2-E: Skill extraction (requires DB)
# ---------------------------------------------------------------------------

@pytest.mark.requires_db
@pytest.mark.requires_seed
class TestSkillExtraction:
    def test_my_resume_extracts_python(self, eval_session):
        from src.services.skill_extractor import extract_skills
        result = _parsed(RESUME_MY)
        skills = extract_skills(result["sections"], result["raw_text"], eval_session)
        skill_names = {s["display_name"].lower() for s in skills}
        assert "python" in skill_names, (
            f"'Python' not extracted from resume. Got: {sorted(skill_names)}"
        )

    def test_my_resume_extracts_fastapi(self, eval_session):
        from src.services.skill_extractor import extract_skills
        result = _parsed(RESUME_MY)
        skills = extract_skills(result["sections"], result["raw_text"], eval_session)
        skill_names = {s["display_name"].lower() for s in skills}
        assert "fastapi" in skill_names, (
            f"'FastAPI' not extracted from resume. Got: {sorted(skill_names)}"
        )

    def test_backend_resume_extracts_postgresql(self, eval_session):
        from src.services.skill_extractor import extract_skills
        result = _parsed(RESUME_BACKEND)
        skills = extract_skills(result["sections"], result["raw_text"], eval_session)
        skill_names = {s["display_name"].lower() for s in skills}
        assert "postgresql" in skill_names, (
            f"'PostgreSQL' not extracted. Got: {sorted(skill_names)}"
        )

    def test_empty_resume_extraction_is_safe(self, eval_session):
        """Extracting from an empty section dict must not crash."""
        from src.services.skill_extractor import extract_skills
        try:
            skills = extract_skills({}, "", eval_session)
            assert isinstance(skills, list)
        except Exception as exc:
            pytest.fail(f"extract_skills on empty input raised: {exc}")

    def test_all_returned_skills_have_required_keys(self, eval_session):
        from src.services.skill_extractor import extract_skills
        result = _parsed(RESUME_ML)
        skills = extract_skills(result["sections"], result["raw_text"], eval_session)
        for s in skills:
            assert "skill_id"     in s, f"Missing skill_id in {s}"
            assert "display_name" in s, f"Missing display_name in {s}"
            assert "confidence"   in s, f"Missing confidence in {s}"
            assert "source"       in s, f"Missing source in {s}"
            assert 0.0 <= s["confidence"] <= 1.0, (
                f"confidence={s['confidence']} out of range for skill {s['display_name']!r}"
            )
