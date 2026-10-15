"""Integration tests for the FastAPI demand forecasting and inventory service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    """GET /health must return 200 OK with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_forecast_endpoint_valid_request() -> None:
    """POST /forecast must return exactly 16 daily predictions with valid confidence bounds."""
    payload = {
        "store_nbr": 1,
        "family": "DAIRY",
        "horizon_days": 16,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["store_nbr"] == 1
    assert data["family"] == "DAIRY"
    assert len(data["predictions"]) == 16
    assert len(data["prediction_lower"]) == 16
    assert len(data["prediction_upper"]) == 16

    # Verify uncertainty bounds: lower <= point <= upper
    for low, point, up in zip(
        data["prediction_lower"], data["predictions"], data["prediction_upper"], strict=True
    ):
        assert low <= point <= up
        assert low >= 0.0


def test_forecast_endpoint_validation_error() -> None:
    """POST /forecast with invalid store number must return 422 Unprocessable Entity."""
    payload = {
        "store_nbr": 999,  # Store number valid range: 1-54
        "family": "GROCERY I",
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 422


def test_optimize_endpoint_valid_request() -> None:
    """POST /optimize must return operational (s, S) parameters and simulation KPIs."""
    payload = {
        "store_nbr": 2,
        "family": "MEATS",
        "service_level": 0.95,
    }
    response = client.post("/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["store_nbr"] == 2
    assert data["family"] == "MEATS"
    assert data["reorder_point_s"] < data["order_up_to_S"]
    assert 0.0 <= data["expected_fill_rate"] <= 1.0
    assert 0.0 <= data["expected_daily_stockout_rate"] <= 1.0
    assert data["expected_daily_holding_cost"] >= 0.0
