import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from api.providers.market_data import PolygonProvider, YahooProvider
from src.shared.schemas.market import MarketQuote

@pytest.mark.asyncio
async def test_polygon_provider_stub():
    with patch("src.shared.utils.http_client.HttpClientManager.get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        
        # Mock last trade response
        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = {"results": {"p": 150.0}}
        
        # Mock prev close response
        mock_resp2 = MagicMock()
        mock_resp2.status_code = 200
        mock_resp2.json.return_value = {"results": [{"c": 145.0}]}
        
        mock_client.get.side_effect = [mock_resp1, mock_resp2]
        
        provider = PolygonProvider(api_key="TEST_KEY")
        result = await provider.get_ticker_data("AAPL")
        
        assert isinstance(result, MarketQuote)
        assert result.symbol == "AAPL"
        assert result.last_price == 150.0
        assert result.provider == "Polygon"

@pytest.mark.asyncio
async def test_yahoo_provider_stub():
    with patch("api.providers.market_data.default_stealth_client") as mock_stealth:
        mock_stealth.get = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "chart": {
                "result": [{
                    "meta": {
                        "regularMarketPrice": 150.0,
                        "chartPreviousClose": 145.0
                    }
                }]
            }
        }
        mock_stealth.get.return_value = mock_resp
        
        provider = YahooProvider()
        result = await provider.get_ticker_data("AAPL")
        
        assert isinstance(result, MarketQuote)
        assert result.symbol == "AAPL"
        assert result.last_price == 150.0
        assert result.provider == "Yahoo"
