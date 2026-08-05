"""Tests for resume_parser.py — pure unit tests, no DB required."""

import io
import pytest
from src.services.resume_parser import parse_resume, ParseError, _detect_sections, get_section_weight


def _make_simple_pdf_bytes() -> bytes:
    """Build a minimal valid PDF in memory."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "SKILLS\nPython FastAPI PostgreSQL Docker\n\nEXPERIENCE\nBuilt REST APIs with FastAPI")
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()
    except ImportError:
        pytest.skip("PyMuPDF not installed")


class TestDetectSections:
    def test_detects_skills_section(self):
        text = "SKILLS\nPython Docker\n\nEXPERIENCE\nBuilt APIs"
        sections = _detect_sections(text)
        assert "skills" in sections or "experience" in sections

    def test_single_block_no_headers(self):
        text = "John Doe, software engineer with 5 years of experience building web apps."
        sections = _detect_sections(text)
        # Should return 'other' or 'summary'
        assert len(sections) >= 1

    def test_empty_text(self):
        sections = _detect_sections("")
        assert sections  # Returns at least {'other': ''}


class TestSectionWeight:
    def test_skills_section_highest_weight(self):
        assert get_section_weight("skills") > get_section_weight("experience")
        assert get_section_weight("skills") > get_section_weight("other")

    def test_unknown_section_default(self):
        assert get_section_weight("random_section") == 1.0

    def test_projects_above_education(self):
        assert get_section_weight("projects") > get_section_weight("education")


class TestParseResume:
    def test_unsupported_extension_raises(self):
        with pytest.raises(ParseError, match="Unsupported file type"):
            parse_resume(b"some bytes", "resume.xlsx")

    def test_empty_file_raises(self):
        with pytest.raises(ParseError):
            parse_resume(b"", "resume.pdf")

    def test_valid_pdf_returns_correct_shape(self):
        pdf_bytes = _make_simple_pdf_bytes()
        result = parse_resume(pdf_bytes, "test.pdf")
        assert "raw_text" in result
        assert "sections" in result
        assert "layout_type" in result
        assert result["layout_type"] == "pdf"
        assert len(result["raw_text"]) > 0
        assert isinstance(result["sections"], dict)

    def test_char_count_matches_raw_text(self):
        pdf_bytes = _make_simple_pdf_bytes()
        result = parse_resume(pdf_bytes, "test.pdf")
        assert result["char_count"] == len(result["raw_text"])
