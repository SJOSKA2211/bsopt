from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ml.scraper import MarketDataScraper


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

@patch("src.ml.scraper.requests.get")
def test_fetch_historical_data_failure(mock_get):
    """Verify that the scraper raises an error after max retries."""
    fail_response = MagicMock()
    fail_response.status_code = 500
    mock_get.return_value = fail_response
    
    scraper = MarketDataScraper(api_key="test_key", provider="auto", max_retries=2)
    
    with pytest.raises(Exception):
        scraper.fetch_historical_data("AAPL", "2023-01-01", "2023-01-03")
    
    # It will fail after trying both AV and Polygon (fallback)
    # AV: 1 + 2 retries = 3
    # Polygon: 1 + 2 retries = 3
    # Total: 6
    assert mock_get.call_count == 6
