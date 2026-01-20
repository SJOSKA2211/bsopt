from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
from src.ml.scraper import MarketDataScraper

@pytest.fixture
def scraper():
    return MarketDataScraper(api_key="DEMO_KEY")

@pytest.fixture
def mock_response():
    mock = MagicMock()
    # Alpha Vantage Format
    mock.json.return_value = {
        "Time Series (Daily)": {
            "2023-01-01": {"1. open": "100.0", "2. high": "105.0", "3. low": "99.0", "4. close": "102.0", "5. volume": "1000"},
            "2023-01-02": {"1. open": "102.0", "2. high": "103.0", "3. low": "101.0", "4. close": "101.5", "5. volume": "800"}
        }
    }
    mock.status_code = 200
    return mock

@patch("src.ml.scraper.requests.get")
@patch("src.ml.scraper.logger")
def test_fetch_historical_data_success(mock_logger, mock_get, mock_response):
    """Verify that the scraper correctly fetches and parses data."""
    mock_get.return_value = mock_response
    
    from src.shared import observability
    with patch.object(observability.SCRAPE_DURATION, 'labels') as mock_duration_labels:
        mock_observe = MagicMock()
        mock_duration_labels.return_value.observe = mock_observe
        
        scraper = MarketDataScraper(api_key="test_key")
        df = scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-03")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert mock_get.called
        assert mock_logger.info.called
        
        # Now expects alpha_vantage
        mock_duration_labels.assert_called_with(api="alpha_vantage")
        assert mock_observe.called

@patch("src.ml.scraper.requests.get")
def test_fetch_historical_data_retry_logic(mock_get, mock_response):
    """Verify that the scraper retries on failure."""
    # Fail twice, then succeed
    fail_response = MagicMock()
    fail_response.status_code = 500
    mock_get.side_effect = [fail_response, fail_response, mock_response]
    
    from src.shared import observability
    with patch.object(observability.SCRAPE_ERRORS, 'labels') as mock_error_labels:
        mock_inc = MagicMock()
        mock_error_labels.return_value.inc = mock_inc
        
        scraper = MarketDataScraper(api_key="test_key", max_retries=3)
        df = scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-03")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert mock_get.call_count == 3
        assert mock_inc.call_count == 2 # 2 failures before success
        # Expect alpha_vantage labels
        mock_error_labels.assert_called_with(api="alpha_vantage", status_code=500)

def test_fetch_historical_data_failure(scraper):
    with patch("src.ml.scraper.requests.get") as mock_get:
        mock_get.return_value.status_code = 500
        with pytest.raises(Exception, match="Failed to fetch data"):
            scraper.fetch_historical_data("AAPL", "2025-01-01", "2025-01-05")

def test_fetch_historical_data_polygon_success(scraper):
    scraper.provider = "polygon"
    with patch("src.ml.scraper.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "status": "OK",
            "results": [
                {"t": 1735689600000, "o": 100.0, "h": 110.0, "l": 90.0, "c": 105.0, "v": 1000}
            ]
        }
        df = scraper.fetch_historical_data("AAPL", "2025-01-01", "2025-01-05")
        assert not df.empty
        assert df.iloc[0]["close"] == 105.0

def test_fetch_historical_data_auto_provider(scraper):
    scraper.provider = "auto"
    with patch("src.ml.scraper.requests.get") as mock_get:
        # First call (Alpha Vantage) fails
        mock_response_av = MagicMock()
        mock_response_av.status_code = 401
        
        # Second call (Polygon) succeeds
        mock_response_poly = MagicMock()
        mock_response_poly.status_code = 200
        mock_response_poly.json.return_value = {
            "status": "OK",
            "results": [{"t": 123, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}]
        }
        
        mock_get.side_effect = [mock_response_av, mock_response_poly]
        
        df = scraper.fetch_historical_data("AAPL", "2025-01-01", "2025-01-05")
        assert not df.empty
        assert scraper.provider == "auto" # Still auto but tried both

def test_scraper_redact_api_key():
    scraper = MarketDataScraper(api_key="SECRET_123")
    redacted = scraper._redact_message("Error with key SECRET_123 in URL")
    assert "SECRET_123" not in redacted
    assert "[REDACTED]" in redacted

def test_scraper_invalid_ticker(scraper):
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        scraper.fetch_historical_data("INVALID T@CKER", "2025-01-01", "2025-01-05")