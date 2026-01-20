import pytest
from strawberry.types import ExecutionResult
from src.pricing.graphql.schema import schema

@pytest.mark.asyncio
async def test_pricing_subgraph_schema_valid():
    """Verify that the schema is valid and has federation support."""
    
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

@pytest.mark.asyncio
async def test_calculate_pricing():
    """Verify calculation of option price and Greeks."""
    
    # We query the _entities field to simulate federation resolving the Option entity
    # But for unit testing, we can just expose a direct query or test the resolver if we make it public.
    # A better way for subgraphs is to test the entity resolution.
    
    query = """
        query {
            _entities(representations: [{ __typename: "Option", id: "AAPL_20260115_C_150", strike: 150.0, underlyingSymbol: "AAPL", expiry: "2026-01-15T00:00:00", optionType: "call" }]) {
                ... on Option {
                    id
                    price
                    delta
                }
            }
        }
    """
    
    # Note: For this to work, the subgraph needs to know how to resolve the reference.
    # We might need to pass enough info in representation or have a lookup.
    # In this simple implementation, we'll calculate based on inputs.
    
    result: ExecutionResult = await schema.execute(query)
    
    assert result.errors is None
    assert result.data is not None
    assert len(result.data["_entities"]) == 1
    option = result.data["_entities"][0]
    assert option["id"] == "AAPL_20260115_C_150"
    assert option["price"] is not None
    assert option["delta"] is not None
