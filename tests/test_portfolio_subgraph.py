import pytest
from strawberry.types import ExecutionResult
from src.portfolio.graphql.schema import schema

@pytest.mark.asyncio
async def test_portfolio_subgraph_schema_valid():
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
async def test_get_portfolio():
    """Verify fetching a portfolio and its positions."""
    
    query = """
        query {
            portfolio(userId: "user_123") {
                id
                userId
                cashBalance
                positions {
                    id
                    contractSymbol
                    quantity
                    entryPrice
                }
            }
        }
    """
    
    result: ExecutionResult = await schema.execute(query)
    
    assert result.errors is None
    assert result.data is not None
    portfolio = result.data["portfolio"]
    assert portfolio["userId"] == "user_123"
    assert len(portfolio["positions"]) > 0
    assert portfolio["positions"][0]["contractSymbol"] == "AAPL_20260115_C_150"
