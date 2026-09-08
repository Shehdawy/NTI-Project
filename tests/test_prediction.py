from fastapi.testclient import TestClient

from backend.main import app


PAYLOAD = {"product_category": "HOBBIES", "date": "2016-05-22", "price": 9.58, "cost_price": 6.23, "competitor_price": 9.87, "inventory": 101, "promotion": True, "season": "Spring", "customer_rating": 4.1}


def test_prediction():
    with TestClient(app) as client:
        response = client.post("/api/v1/predict-demand", json=PAYLOAD)
        assert response.status_code == 200
        assert response.json()["predicted_demand"] >= 0


def test_invalid_price():
    invalid = {**PAYLOAD, "price": 0}
    with TestClient(app) as client:
        assert client.post("/api/v1/predict-demand", json=invalid).status_code == 422