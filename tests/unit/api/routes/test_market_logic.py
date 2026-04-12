from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.index import app
from src.shared.schemas.market import MarketQuote

client = TestClient(app)

@pytest.fixture
def mock_market_user():
    from src.auth.auth import get_current_active_user
    mock_user = MagicMock()
    mock_user.id = "test-user-id"
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    yield mock_user
    app.dependency_overrides.clear()

@pytest.fixture
def mock_market_router():
    with patch("api.routes.market.market_router_engine") as mock_engine:
        yield mock_engine

def test_get_tickers_success(mock_market_user, mock_market_router):
    # Setup mock return value
    mock_quote = MagicMock(spec=MarketQuote)
    mock_quote.to_ticker.return_value = {"symbol": "AAPL", "price": 150.0}
    mock_market_router.get_live_quote = AsyncMock(return_value=mock_quote)
    
    with patch("src.shared.config.settings") as mock_settings:
        mock_settings.MARKET_TICKER_SYMBOLS = ["AAPL"]
        
        response = client.get("/api/v1/market/tickers")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "AAPL"

def test_get_tickers_failure_fallback(mock_market_user, mock_market_router):
    # Simulate an exception in the engine
    mock_market_router.get_live_quote.side_effect = Exception("Data source down")
    
    with patch("src.shared.config.settings") as mock_settings:
        mock_settings.MARKET_TICKER_SYMBOLS = ["AAPL"]
        
        response = client.get("/api/v1/market/tickers")
        
        # Route is designed to return empty list on failure (per lines 43-45)
        assert response.status_code == 200
        assert response.json() == []

@pytest.mark.asyncio
async def test_sse_market_data_connection(mock_market_router):
    # SSE tests are harder with TestClient as it's synchronous
    # But we can verify the response type and headers
    mock_quote = MagicMock(spec=MarketQuote)
    mock_quote.to_ticker.return_value = {"symbol": "BTC", "price": 50000.0}
    mock_market_router.get_live_quote = AsyncMock(return_value=mock_quote)
    
    with TestClient(app) as test_client:
        # We use a timeout to not get stuck in the infinite loop of the generator
        # However, StreamingResponse with infinite loop might need special handling
        # or we just check the start of the stream
        response = test_client.get("/api/v1/market/sse/market-data?symbols=BTC")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
