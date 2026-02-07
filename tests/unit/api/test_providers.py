import pytest

from src.api.providers.market_data import PolygonProvider, YahooProvider


@pytest.mark.asyncio
async def test_polygon_provider_stub():
    provider = PolygonProvider()
    result = await provider.get_ticker_data("AAPL")
    assert result["symbol"] == "AAPL"
    assert result["provider"] == "Polygon"

@pytest.mark.asyncio
async def test_yahoo_provider_stub():
    provider = YahooProvider()
    result = await provider.get_ticker_data("AAPL")
    assert result["symbol"] == "AAPL"
    assert result["provider"] == "Yahoo"
