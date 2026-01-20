import pytest
from fastapi.testclient import TestClient
from src.streaming.main import app
import json
import asyncio

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_websocket_marketdata():
    with client.websocket_connect("/marketdata") as websocket:
        # It sends data immediately
        data = websocket.receive_text()
        obj = json.loads(data)
        assert obj["symbol"] == "AAPL"
        assert "close" in obj

def test_graphql_endpoint():
    # Simple introspection query or a basic query
    query = '{ __typename }'
    response = client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    assert response.json()["data"]["__typename"] == "Query"
