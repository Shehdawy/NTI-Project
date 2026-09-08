from fastapi.testclient import TestClient

from backend.main import app


PAYLOAD = {"product_category": "HOBBIES", "date": "2016-05-22", "current_price": 9.58, "cost_price": 6.23, "competitor_price": 9.87, "inventory": 101, "promotion": True, "season": "Spring", "customer_rating": 4.1, "objective": "maximize_profit"}


def test_recommendation():
    with TestClient(app) as client:
        response = client.post("/api/v1/recommend-price", json=PAYLOAD)
        body = response.json()
        assert response.status_code == 200
        assert len(body["scenarios"]) == 25
        assert body["recommendation"]["expected_profit"] is not None
        assert body["recommendation"]["recommended_price"] > 0


def test_invalid_objective():
    with TestClient(app) as client:
        response = client.post("/api/v1/recommend-price", json={**PAYLOAD, "objective": "random"})
        assert response.status_code == 422