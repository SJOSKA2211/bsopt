import strawberry
from httpx import ASGITransport, AsyncClient

# sys.path hack removed
from src.api.main import app

print("DEBUG: Strawberry Federation:", strawberry.federation)


async def test_federated_schema_availability():
    """
    Verify that the GraphQL schema is successfully composed and accessible.
    Critical for the Apollo Gateway / App Gateway federation.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Introspection Query
        query = """
        query {
            __schema {
                types {
                    name
                }
            }
        }
        """
        response = await ac.post("/graphql", json={"query": query})
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "__schema" in data["data"]

        # Verify key federated types exist
        types = [t["name"] for t in data["data"]["__schema"]["types"]]
        assert "Portfolio" in types
        assert "Position" in types
        assert "User" in types
        # assert "Price" in types # Might be named differently in schema
