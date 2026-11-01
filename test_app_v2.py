
from fastapi.testclient import TestClient
from app_v2 import app

client = TestClient(app)

# Test 1: Homepage works
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["threshold"] == 0.4

# Test 2: Prediction endpoint works
def test_prediction():
    response = client.post(
        "/predict",
        json={"text": "The acting was amazing."}
    )

    assert response.status_code == 200

    data = response.json()

    assert "is_spoiler" in data
    assert "spoiler_probability" in data
    assert data["threshold"] == 0.4

# Test 3: Prediction follows our threshold
def test_threshold():
    response = client.post(
        "/predict",
        json={"text": "The ending was unexpected."}
    )

    data = response.json()

    assert data["is_spoiler"] == (
        data["spoiler_probability"] >= data["threshold"]
    )
