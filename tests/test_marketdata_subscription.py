import pytest

from src.streaming.graphql.schema import schema


@pytest.mark.asyncio
async def test_market_data_subscription():
    """Verify market data subscription."""

    query = """
        subscription {
            marketDataStream(symbols: ["AAPL"]) {
                symbol
                lastPrice
            }
        }
    """

    # Execute subscription
    # strawberry schema.subscribe() returns an async generator
    sub = await schema.subscribe(query)

    # Get first result
    result = await anext(sub)

    assert result.errors is None
    assert result.data is not None
    assert result.data["marketDataStream"]["symbol"] == "AAPL"
    assert result.data["marketDataStream"]["lastPrice"] > 0
