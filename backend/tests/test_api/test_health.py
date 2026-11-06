"""
API smoke tests. conftest.py points Qdrant at a dead port and turns the
reranker off, so startup downloads nothing and the vector DB is unreachable.
"""

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_health_reports_degraded_without_qdrant():
    with TestClient(create_app()) as client:
        resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant_connected"] is False
    assert "General" in body["domain_model_map"]


def test_openapi_lists_core_routes():
    with TestClient(create_app()) as client:
        paths = client.get("/openapi.json").json()["paths"]
    assert "/health" in paths
    assert any(p.startswith("/api/v1/") for p in paths)
