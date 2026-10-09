
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Spoiler Alert API is running!"

def test_prediction():
    response = client.post(
        "/predict",
        json={"text": "The acting was amazing."}
    )

    assert response.status_code == 200

    result = response.json()
    assert "is_spoiler" in result
    assert "spoiler_probability" in result
    assert isinstance(result["is_spoiler"], bool)
    assert 0 <= result["spoiler_probability"] <= 1

def test_monitoring():
    response = client.get("/stats")

    assert response.status_code == 200

    result = response.json()
    assert "total_requests" in result
    assert "prediction_errors" in result
