"""Tests for the resume upload endpoint."""

import io
import pytest


def _make_pdf_bytes() -> bytes:
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Python Developer with FastAPI experience")
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()
    except ImportError:
        pytest.skip("PyMuPDF not installed")


class TestUploadEndpoint:
    def test_missing_file_returns_422(self, client):
        resp = client.post("/v1/resume/upload", data={"target_role": "Backend Engineer"})
        assert resp.status_code == 422

    def test_unsupported_file_type_returns_422(self, client):
        resp = client.post(
            "/v1/resume/upload",
            files={"file": ("resume.xlsx", b"fake bytes", "application/vnd.ms-excel")},
            data={"target_role": "Backend Engineer"},
        )
        assert resp.status_code == 422

    def test_empty_role_returns_422(self, client):
        pdf_bytes = _make_pdf_bytes()
        resp = client.post(
            "/v1/resume/upload",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
            data={"target_role": ""},
        )
        assert resp.status_code in (422, 400)

    def test_valid_upload_returns_202(self, client, monkeypatch):
        # Mock the Celery task so we don't need a running worker
        import src.tasks.analyze_resume as task_module

        class FakeAsyncResult:
            id = "fake-celery-id"

        def mock_apply_async(**kwargs):
            return FakeAsyncResult()

        monkeypatch.setattr(task_module.analyze_resume_task, "apply_async", mock_apply_async)

        pdf_bytes = _make_pdf_bytes()
        resp = client.post(
            "/v1/resume/upload",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
            data={"target_role": "Backend Software Engineer", "duration": "3 months"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "session_id" in data
        assert "processing_url" in data
        assert data["status"] == "queued"
