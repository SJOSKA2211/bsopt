import httpx
import pytest

from src.shared.config import settings


@pytest.mark.asyncio
async def test_ml_comparison_endpoint():
    """Verify that /ml/comparison returns structured data without mocks."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # 1. Fetch comparison metrics
        response = await client.get("/ml/comparison")
        assert response.status_code == 200
        
        data = response.json()
        assert "userPnl" in data
        assert "aiPnl" in data
        assert "winRate" in data
        assert isinstance(data["userPnl"], (int, float))

@pytest.mark.asyncio
async def test_ml_predictions_endpoint():
    """Verify that /ml/predictions returns valid schema."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/ml/predictions")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            prediction = data[0]
            assert "symbol" in prediction
            assert "predicted_price" in prediction
            assert prediction["symbol"] in settings.MARKET_TICKER_SYMBOLS
