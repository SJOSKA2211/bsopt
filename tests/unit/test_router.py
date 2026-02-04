import pytest
from unittest.mock import AsyncMock, patch
from src.data.router import MarketDataRouter

@pytest.mark.asyncio
async def test_router_nse_suffix():
    with patch("src.data.router.NSEScraper") as mock_nse:
        mock_nse.return_value.get_ticker_data = AsyncMock(return_value={"price": 10.0})
        router = MarketDataRouter()
        
        result = await router.get_live_quote("SCOM.NR")
        assert result["price"] == 10.0
        mock_nse.return_value.get_ticker_data.assert_called_with("SCOM")

@pytest.mark.asyncio
async def test_router_nse_market_flag():
    with patch("src.data.router.NSEScraper") as mock_nse:
        mock_nse.return_value.get_ticker_data = AsyncMock(return_value={"price": 10.0})
        router = MarketDataRouter()
        
        result = await router.get_live_quote("SCOM", market="NSE")
        assert result["price"] == 10.0
        mock_nse.return_value.get_ticker_data.assert_called_with("SCOM")

@pytest.mark.asyncio
async def test_router_crypto_detection():
    # This is a placeholder as CCXT logic is omitted in the snippet but mentioned in PRD
    with patch("src.data.router.PolygonProvider") as mock_poly:
        mock_poly.return_value.get_ticker_data = AsyncMock(return_value={"price": 50000.0})
        router = MarketDataRouter()
        result = await router.get_live_quote("BTC-USD")
        assert result["price"] == 50000.0

@pytest.mark.asyncio
async def test_router_polygon_success():
    with patch("src.data.router.PolygonProvider") as mock_poly:
        mock_poly.return_value.get_ticker_data = AsyncMock(return_value={"price": 150.0})
        router = MarketDataRouter()
        
        result = await router.get_live_quote("AAPL")
        assert result["price"] == 150.0
        mock_poly.return_value.get_ticker_data.assert_called_with("AAPL")

@pytest.mark.asyncio
async def test_router_fallback_to_yahoo():
    with patch("src.data.router.PolygonProvider") as mock_poly:
        with patch("src.data.router.YahooProvider") as mock_yahoo:
            mock_poly.return_value.get_ticker_data = AsyncMock(side_effect=Exception("API limit"))
            mock_yahoo.return_value.get_ticker_data = AsyncMock(return_value={"price": 155.0})
            
            router = MarketDataRouter()
            result = await router.get_live_quote("AAPL")
            
            assert result["price"] == 155.0
            mock_yahoo.return_value.get_ticker_data.assert_called_with("AAPL")