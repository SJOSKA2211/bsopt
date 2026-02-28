from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_routes():
    response = client.get("/api/v1/auth/jwks")
    print(response.status_code)
    print(response.json())

test_routes()
