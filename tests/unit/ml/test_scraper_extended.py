from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ml.scraper import MarketDataScraper


@pytest.fixture
def scraper():
    return MarketDataScraper(api_key="test_key", provider="alpha_vantage")

@pytest.fixture
def mock_client():
    with patch("src.utils.http_client.HttpClientManager.get_client") as mock_get:
        client = AsyncMock()
        mock_get.return_value = client
        yield client

def test_validate_inputs(scraper):
    scraper._validate_inputs("AAPL", "2023-01-01", "2023-01-31")
    with pytest.raises(ValueError):
        scraper._validate_inputs("INVALID$", "2023-01-01", "2023-01-31")

def test_redact_message(scraper):
    msg = "Error with key test_key at URL"
    assert scraper._redact_message(msg) == "Error with key [REDACTED] at URL"

@pytest.mark.asyncio
async def test_fetch_historical_data_alpha_vantage_success(scraper, mock_client):
    mock_client.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"Time Series (Daily)": {"2023-01-01": {"1. open": "100.0", "2. high": "105.0", "3. low": "95.0", "4. close": "102.0", "5. volume": "1000"}}}
    )
    df = await scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-01")
    assert not df.empty
    assert df.iloc[0]["close"] == 102.0

@pytest.mark.asyncio
async def test_fetch_historical_data_polygon_success(scraper, mock_client):
    scraper.provider = "polygon"
    mock_client.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"status": "OK", "results": [{"t": 1672531200000, "o": 100.0, "h": 105.0, "l": 95.0, "c": 102.0, "v": 1000}]}
    )
    df = await scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-01")
    assert not df.empty
    assert df.iloc[0]["close"] == 102.0

@pytest.mark.asyncio
async def test_fetch_historical_data_alpha_vantage_rate_limit(scraper, mock_client):
    mock_client.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"Note": "rate limit reached"}
    )
    with patch("asyncio.sleep"): # Skip sleep
        with pytest.raises(Exception, match="rate limit reached"):
            await scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-01")

@pytest.mark.asyncio
async def test_fetch_historical_data_auto_fallback(mock_client):
    scraper = MarketDataScraper(api_key="test_key", provider="auto", max_retries=0)
    
    # First call (Alpha Vantage) fails with 401
    mock_response_av = MagicMock()
    mock_response_av.status_code = 401
    
    # Second call (Polygon) succeeds
    mock_response_poly = MagicMock()
    mock_response_poly.status_code = 200
    mock_response_poly.json.return_value = {
        "status": "OK", 
        "results": [{"t": 1672531200000, "o": 100.0, "h": 105.0, "l": 95.0, "c": 102.0, "v": 1000}]
    }
    
    mock_client.get.side_effect = [mock_response_av, mock_response_poly]
    
    df = await scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-01")
    assert not df.empty
    assert df.iloc[0]["close"] == 102.0
    assert mock_client.get.call_count == 2
