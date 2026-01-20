import pytest
from fastapi.testclient import TestClient
from src.pricing.main import app
from datetime import datetime, timedelta

def test_pricing_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_pricing_graphql_dummy():
    client = TestClient(app)
    # Strawberry strips leading underscores? Or we should use camelCase.
    # Let's try 'dummy'
    query = "{ dummy }"
    response = client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    assert response.json()["data"]["dummy"] == "pricing"

@pytest.mark.asyncio
async def test_pricing_graphql_option_reference():
    client = TestClient(app)
    expiry = (datetime.now() + timedelta(days=365)).isoformat()
    query = """
        query($rep: [_Any!]!) {
            _entities(representations: $rep) {
                ... on Option {
                    price
                    delta
                    gamma
                }
            }
        }
    """
    rep = [{
        "__typename": "Option",
        "id": "test_opt",
        "strike": 100.0,
        "underlyingSymbol": "AAPL",
        "expiry": expiry,
        "optionType": "call"
    }]
    response = client.post("/graphql", json={"query": query, "variables": {"rep": rep}})
    assert response.status_code == 200
    data = response.json()["data"]["_entities"][0]
    assert data["price"] > 0
    assert data["delta"] > 0
    assert data["gamma"] > 0