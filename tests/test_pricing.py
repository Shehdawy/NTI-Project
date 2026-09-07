from fastapi.testclient import TestClient

from backend.main import app


PAYLOAD = {"product_category": "Home", "date": "2024-12-31", "current_price": 164.36, "cost_price": 86.77, "competitor_price": 149.54, "inventory": 101, "promotion": True, "season": "Winter", "customer_rating": 4.4, "objective": "maximize_profit"}


def test_recommendation():
    with TestClient(app) as client:
        response = client.post("/api/v1/recommend-price", json=PAYLOAD)
        body = response.json()
        assert response.status_code == 200
        assert len(body["scenarios"]) == 25
        assert body["recommendation"]["expected_profit"] is not None


def test_invalid_objective():
    with TestClient(app) as client:
        response = client.post("/api/v1/recommend-price", json={**PAYLOAD, "objective": "random"})
        assert response.status_code == 422