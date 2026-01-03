from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import requests
from src.ml.scraper import MarketDataScraper

@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.json.return_value = {
        "results": [
            {"t": 1672531200000, "o": 100.0, "h": 105.0, "l": 99.0, "c": 102.0, "v": 1000},
            {"t": 1672617600000, "o": 102.0, "h": 103.0, "l": 101.0, "c": 101.5, "v": 800}
        ],
        "status": "OK"
    }
    mock.status_code = 200
    return mock

@patch("src.ml.scraper.requests.get")
def test_fetch_historical_data_success(mock_get, mock_response):
    """Verify that the scraper correctly fetches and parses data."""
    mock_get.return_value = mock_response
    
    scraper = MarketDataScraper(api_key="test_key")
    df = scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-03")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df.iloc[0]["open"] == 100.0
    
    mock_get.assert_called_once()

@patch("src.ml.scraper.requests.get")
def test_fetch_historical_data_retry_logic(mock_get, mock_response):
    """Verify that the scraper retries on failure."""
    # Fail twice, then succeed
    fail_response = MagicMock()
    fail_response.status_code = 500
    mock_get.side_effect = [fail_response, fail_response, mock_response]
    
    scraper = MarketDataScraper(api_key="test_key", max_retries=3)
    df = scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-03")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert mock_get.call_count == 3

@patch("src.ml.scraper.requests.get")
def test_fetch_historical_data_failure(mock_get):
    """Verify that the scraper raises an error after max retries."""
    fail_response = MagicMock()
    fail_response.status_code = 500
    mock_get.return_value = fail_response
    
    scraper = MarketDataScraper(api_key="test_key", max_retries=2)
    
    with pytest.raises(Exception) as excinfo:
        scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-03")
    
    assert "Failed to fetch data" in str(excinfo.value)
    assert mock_get.call_count == 3 # Initial + 2 retries
