from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_ping():
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_returns_number():
    payload = {
        "vendor_id": 2,
        "passenger_count": 1,
        "trip_distance": 3.2,
        "rate_code": 1,
        "payment_type": 1,
        "pickup_hour": 14,
        "pickup_weekday": 3,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_duration_min" in body
    assert isinstance(body["predicted_duration_min"], (int, float))
