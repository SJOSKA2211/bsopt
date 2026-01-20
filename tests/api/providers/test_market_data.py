import pytest
from src.api.providers.market_data import PolygonProvider, YahooProvider

@pytest.mark.asyncio
async def test_polygon_provider():
    provider = PolygonProvider()
    
    data = await provider.get_ticker_data("AAPL")
    assert data["symbol"] == "AAPL"
    assert data["provider"] == "Polygon"
    
    search_results = await provider.search("AAPL")
    assert len(search_results) > 0
    assert search_results[0]["symbol"] == "AAPL"
    
    chain = await provider.get_option_chain("AAPL")
    assert len(chain) > 0
    assert chain[0]["symbol"] == "AAPL"

@pytest.mark.asyncio
async def test_yahoo_provider():
    provider = YahooProvider()
    
    data = await provider.get_ticker_data("GOOGL")
    assert data["symbol"] == "GOOGL"
    assert data["provider"] == "Yahoo"
    
    search_results = await provider.search("GOOGL")
    assert len(search_results) > 0
    assert search_results[0]["symbol"] == "GOOGL"
    
    chain = await provider.get_option_chain("GOOGL")
    assert len(chain) > 0
    assert chain[0]["symbol"] == "GOOGL"
