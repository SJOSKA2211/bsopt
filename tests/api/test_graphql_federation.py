import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from src.api.main import app
from src.api.main import GraphQLRouter as OriginalGraphQLRouter # Import original for patching

@pytest.fixture(autouse=True)
def patch_graphql_router():
    """
    Patches the GraphQLRouter to remove context_getter for tests.
    """
    with patch('src.api.main.GraphQLRouter') as MockGraphQLRouter:
        # Mimic the constructor but without context_getter
        def mock_init(schema, graphiql=False, *args, **kwargs):
            return OriginalGraphQLRouter(schema, graphiql=graphiql, *args, **kwargs)
        MockGraphQLRouter.side_effect = mock_init
        yield # Correctly import app from src.api.main 


@pytest.mark.asyncio
async def test_graphql_query():
    """
    Test a simple GraphQL query to ensure the API is operational.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/graphql", json={"query": "{ option(contractSymbol: \"SPY_CALL_400\") { id contractSymbol } }"})
        assert response.status_code == 200
        assert response.json()["data"]["option"]["id"] == "1"
        assert response.json()["data"]["option"]["contractSymbol"] == "SPY_CALL_400"

@pytest.mark.asyncio
async def test_graphql_mutation():
    """
    Test a simple GraphQL mutation.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/graphql", json={
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
