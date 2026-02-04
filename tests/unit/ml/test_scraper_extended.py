import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.ml.scraper import MarketDataScraper

@pytest.fixture
def scraper():
    return MarketDataScraper(api_key="test_key")

def test_validate_inputs(scraper):
    scraper._validate_inputs("AAPL", "2023-01-01", "2023-01-31")
    
    with pytest.raises(ValueError):
        scraper._validate_inputs("INVALID$", "2023-01-01", "2023-01-31")
    
    with pytest.raises(ValueError):
        scraper._validate_inputs("AAPL", "01-01-2023", "2023-01-31")

def test_redact_message(scraper):
    msg = "Error with key test_key at URL"
    assert scraper._redact_message(msg) == "Error with key [REDACTED] at URL"
    
    demo_scraper = MarketDataScraper(api_key="DEMO_KEY")
    assert demo_scraper._redact_message(msg) == msg

@patch("src.ml.scraper.requests.get")
def test_fetch_historical_data_alpha_vantage_success(mock_get, scraper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "Time Series (Daily)": {
            "2023-01-01": {"1. open": "100.0", "2. high": "105.0", "3. low": "95.0", "4. close": "102.0", "5. volume": "1000"}
        }
    }
    mock_get.return_value = mock_response
    
    df = scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-01")
    assert not df.empty
    assert df.iloc[0]["close"] == 102.0

@patch("src.ml.scraper.requests.get")
def test_fetch_historical_data_polygon_success(mock_get):
    scraper = MarketDataScraper(api_key="test_key", provider="polygon")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "OK",
        "results": [
            {"t": 1672531200000, "o": 100.0, "h": 105.0, "l": 95.0, "c": 102.0, "v": 1000}
        ]
    }
    mock_get.return_value = mock_response
    
    df = scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-01")
    assert not df.empty
    assert df.iloc[0]["close"] == 102.0

@patch("src.ml.scraper.requests.get")
def test_fetch_historical_data_alpha_vantage_rate_limit(mock_get, scraper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 5 calls per minute"}
    mock_get.return_value = mock_response
    
    with patch("time.sleep"): # Skip sleep
        with pytest.raises(Exception, match="Alpha Vantage rate limit reached"):
            scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-01")

@patch("src.ml.scraper.requests.get")
def test_fetch_historical_data_auto_fallback(mock_get):
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
    
    mock_get.side_effect = [mock_response_av, mock_response_poly]
    
    df = scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-01")
    assert not df.empty
    assert df.iloc[0]["close"] == 102.0
    assert mock_get.call_count == 2
