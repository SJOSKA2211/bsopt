import os
import sys
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.api.main import GraphQLRouter as OriginalGraphQLRouter  # Import original for patching
from src.api.main import app


@pytest.fixture(autouse=True)
def patch_graphql_router():
    """
    Patches the GraphQLRouter to remove context_getter for tests.
    Also mocks OPAEnforcer for authorization checks.
    """
    with patch('src.api.main.GraphQLRouter') as MockGraphQLRouter, \
         patch('src.shared.security.OPAEnforcer.is_authorized', return_value=True):
        # Mimic the constructor but without context_getter
        def mock_init(schema, graphql_ide=False, *args, **kwargs):
            return OriginalGraphQLRouter(schema, graphql_ide=graphql_ide, *args, **kwargs)
        MockGraphQLRouter.side_effect = mock_init
        yield # Correctly import app from src.api.main 


@pytest.mark.asyncio
async def test_graphql_query():
    """
    Test a simple GraphQL query to ensure the API is operational.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers = {
            "X-SSL-Client-Verify": "SUCCESS",
            "X-SSL-Client-S-DN": "CN=test-client",
            "X-User-Id": "test_user",
            "X-User-Role": "admin" # Or 'trader' if appropriate for 'read' on 'options'
        }
        response = await ac.post("/graphql", headers=headers, json={"query": "{ option(contractSymbol: \"SPY_CALL_400\") { id contractSymbol } }"})
        assert response.status_code == 200
        assert response.json()["data"]["option"]["id"] == "SPY_CALL_400" # Expecting contract_symbol as ID
        assert response.json()["data"]["option"]["contractSymbol"] == "SPY_CALL_400"

@pytest.mark.asyncio
async def test_graphql_mutation():
    """
    Test a simple GraphQL mutation.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers = {
            "X-SSL-Client-Verify": "SUCCESS",
            "X-SSL-Client-S-DN": "CN=test-client",
            "X-User-Id": "test_user",
            "X-User-Role": "admin" # Admin can perform mutations
        }
        response = await ac.post("/graphql", headers=headers, json={
            "query": """
                mutation {
                    createPortfolio(userId: \"user1\", name: \"Test Portfolio\", initialCash: 10000.0) {
                        id
                        name
                        cashBalance
                    }
                }
            """
        })
        assert response.status_code == 200
        assert response.json()["data"]["createPortfolio"]["name"] == "Test Portfolio"
        assert response.json()["data"]["createPortfolio"]["cashBalance"] == 10000.0
