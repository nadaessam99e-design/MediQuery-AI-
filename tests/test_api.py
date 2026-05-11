from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    # بيتأكد إن الـ API شغال
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health():
    # بيتأكد إن الـ health check شغال
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ask_valid_question():
    # بيتأكد إن سؤال حقيقي بيرجع إجابة
    response = client.post("/ask", json={"question": "what is ibuprofen?"})
    assert response.status_code == 200
    assert "answer" in response.json()

def test_ask_empty_question():
    # بيتأكد إن سؤال فاضي بيرجع error
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400
