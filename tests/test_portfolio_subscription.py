import pytest

from src.portfolio.graphql.schema import schema


@pytest.mark.asyncio
async def test_portfolio_subscription():
    """Verify portfolio subscription."""

    query = """
        subscription {
            portfolioUpdates(portfolioId: "port_123") {
                id
                cashBalance
            }
        }
    """

    sub = await schema.subscribe(query)
    result = await anext(sub)

    assert result.errors is None
    assert result.data is not None
    assert result.data["portfolioUpdates"]["id"] == "port_123"
    assert result.data["portfolioUpdates"]["cashBalance"] > 0
