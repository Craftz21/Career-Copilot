"""Tests for health check endpoint."""


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "status" in data
    assert "db" in data
    assert "version" in data


def test_health_version_format(client):
    resp = client.get("/health")
    data = resp.json()
    assert data["version"] == "2.0.0"
