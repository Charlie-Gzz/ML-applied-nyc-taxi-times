from fastapi.testclient import TestClient
from src.app.main import app

def test_ping():
    client = TestClient(app)
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
