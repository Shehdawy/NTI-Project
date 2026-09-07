from fastapi.testclient import TestClient

from backend.main import app


PAYLOAD = {"product_category": "Home", "date": "2024-12-31", "price": 164.36, "cost_price": 86.77, "competitor_price": 149.54, "inventory": 101, "promotion": True, "season": "Winter", "customer_rating": 4.4}


def test_prediction():
    with TestClient(app) as client:
        response = client.post("/api/v1/predict-demand", json=PAYLOAD)
        assert response.status_code == 200
        assert response.json()["predicted_demand"] >= 0


def test_invalid_price():
    invalid = {**PAYLOAD, "price": 0}
    with TestClient(app) as client:
        assert client.post("/api/v1/predict-demand", json=invalid).status_code == 422