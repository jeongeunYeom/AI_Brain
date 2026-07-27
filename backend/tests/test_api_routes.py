from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_refactored_routes_are_registered() -> None:
    paths = {route.path for route in app.routes}

    assert "/api/figures/{filename}" in paths
    assert "/api/figure-previews/{filename}" in paths
    assert "/api/evaluation/runs" in paths
    assert "/api/evaluation/latest" in paths
    assert "/api/evaluation/runs/{run_id}" in paths


def test_invalid_evaluation_run_id_returns_404() -> None:
    response = client.get("/api/evaluation/runs/not-a-valid-run-id")

    assert response.status_code == 404
    assert "유효하지 않은 benchmark run ID" in response.json()["detail"]


def test_unsupported_figure_extension_returns_404() -> None:
    response = client.get("/api/figures/not-an-image.txt")

    assert response.status_code == 404
