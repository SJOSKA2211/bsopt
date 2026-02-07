import pytest
from strawberry.types import ExecutionResult

from src.api.graphql.options import schema


@pytest.mark.asyncio
async def test_options_subgraph_schema_valid():
    """Verify that the schema is valid and can execute a simple query."""

    query = """
        query {
            _service {
                sdl
            }
        }
    """

    result: ExecutionResult = await schema.execute(query)

    assert result.errors is None
    assert result.data is not None
    assert "_service" in result.data
    assert "sdl" in result.data["_service"]


@pytest.mark.asyncio
async def test_resolve_option():
    """Verify that we can query an Option by id (or other unique identifier)."""

    query = """
        query {
            option(contractSymbol: "AAPL_20260115_C_150") {
                id
                contractSymbol
                underlyingSymbol
                strike
                expiry
                optionType
            }
        }
    """

    result: ExecutionResult = await schema.execute(query)

    assert result.errors is None
    assert result.data is not None
    assert result.data["option"]["contractSymbol"] == "AAPL_20260115_C_150"
    assert result.data["option"]["underlyingSymbol"] == "AAPL"


def test_graphql_endpoint():
    """Verify that the /graphql endpoint is mounted and working in the FastAPI app."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)

    query = """
        query {
            option(contractSymbol: "AAPL_20260115_C_150") {
                id
            }
        }
    """

    headers = {
        "X-SSL-Client-Verify": "SUCCESS",
        "X-SSL-Client-S-DN": "CN=test-client",
        "X-User-Id": "test_user",
        "X-User-Role": "admin",
    }

    with patch("src.shared.security.OPAEnforcer.is_authorized", return_value=True):
        response = client.post("/graphql", headers=headers, json={"query": query})

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["option"]["id"] == "AAPL_20260115_C_150"
