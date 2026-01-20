import pytest
from fastapi.testclient import TestClient
from src.pricing.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_graphql_endpoint_exists():
    # Just check if it responds (could be 405 or 200 depending on GET/POST)
    response = client.get("/graphql")
    # Strawberry GraphQL usually allows GET for the playground
    assert response.status_code in [200, 405]
