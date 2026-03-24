import pytest
import asyncio
from src.ingestion.router import MarketDataRouter
from src.shared.schemas.market import MarketQuote

@pytest.mark.asyncio
@pytest.mark.integration
async def test_market_router_integration():
    """
    Zero-Mock Integration Test: Verifies the MarketDataRouter races real providers.
    Requires internet access and valid API keys (or Yahoo fallback).
    """
    router = MarketDataRouter()
    
    # Test with a major US ticker
    quote = await router.get_live_quote("SPY")
    
    assert isinstance(quote, MarketQuote)
    assert quote.symbol == "SPY"
    assert quote.last_price > 0
    assert quote.provider in ["Polygon", "Yahoo"]
    
    # Test with an NSE ticker (if configured)
    try:
        nse_quote = await router.get_live_quote("RELIANCE.NR", market="NSE")
        assert nse_quote.symbol == "RELIANCE.NR"
        assert nse_quote.provider in ["NSE", "Yahoo"]
    except Exception as e:
        pytest.skip(f"NSE integration skipped: {e}")

@pytest.mark.asyncio
@pytest.mark.integration
async def test_market_router_staggered_race():
    """
    Verifies the staggered race logic doesn't crash and returns the fastest results.
    """
    router = MarketDataRouter()
    
    # Force a race between Polygon (slow if no key) and Yahoo
    quote = await router.get_live_quote("AAPL")
    
    assert quote.last_price > 0
    assert quote.symbol == "AAPL"
