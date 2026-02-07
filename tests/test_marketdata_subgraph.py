import pytest
from strawberry.types import ExecutionResult

from src.streaming.graphql.schema import schema


@pytest.mark.asyncio
async def test_marketdata_subgraph_schema_valid():
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
async def test_market_data_fields():
    """Verify Market Data fields on Option entity."""
    
    query = """
        query {
            _entities(representations: [{ __typename: "Option", id: "AAPL_20260115_C_150" }]) {
                ... on Option {
                    id
                    lastPrice
                    volume
                }
            }
        }
    """
    
    result: ExecutionResult = await schema.execute(query)
    
    assert result.errors is None
    assert result.data is not None
    assert len(result.data["_entities"]) == 1
    option = result.data["_entities"][0]
    assert option["id"] == "AAPL_20260115_C_150"
    assert option["lastPrice"] is not None
    assert option["volume"] is not None
