import time
import requests
import pandas as pd
import structlog
from typing import Optional
from src.shared.observability import SCRAPE_DURATION, SCRAPE_ERRORS

logger = structlog.get_logger()

class MarketDataScraper:
    """
    Scraper for fetching market data from external APIs (e.g., Polygon.io, Yahoo Finance).
    Includes retry logic and error handling.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io", max_retries: int = 3):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.api_name = "polygon" if "polygon" in base_url else "unknown"

    def fetch_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a given ticker.
        
        Args:
            ticker: The stock symbol (e.g., 'AAPL').
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            
        Returns:
            pd.DataFrame: DataFrame containing timestamp, open, high, low, close, volume.
            
        Raises:
            Exception: If data cannot be fetched after max retries.
        """
        # Polygon.io endpoint for aggregates (bars)
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {"apiKey": self.api_key}
        
        start_time = time.time()
        for attempt in range(self.max_retries + 1): # Try once, then retry max_retries times
            try:
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "OK" and "results" in data:
                        df = pd.DataFrame(data["results"])
                        # Rename columns to standard names
                        df = df.rename(columns={
                            "t": "timestamp",
                            "o": "open",
                            "h": "high",
                            "l": "low",
                            "c": "close",
                            "v": "volume"
                        })
                        
                        duration = time.time() - start_time
                        SCRAPE_DURATION.labels(api=self.api_name).observe(duration)
                        logger.info("scrape_success", ticker=ticker, duration=duration, attempt=attempt)
                        
                        # Convert timestamp to datetime if needed, but keeping as int for now per tests
                        return df[["timestamp", "open", "high", "low", "close", "volume"]]
                    else:
                        logger.warning("scrape_api_error", ticker=ticker, status=data.get("status"), attempt=attempt)
                else:
                    SCRAPE_ERRORS.labels(api=self.api_name, status_code=response.status_code).inc()
                    logger.warning("scrape_http_error", ticker=ticker, status_code=response.status_code, attempt=attempt)
                
                # If we are here, something went wrong (non-200 or invalid data)
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt)) # Exponential backoff
                    continue
                
            except requests.RequestException as e:
                SCRAPE_ERRORS.labels(api=self.api_name, status_code="exception").inc()
                logger.error("scrape_exception", ticker=ticker, error=str(e), attempt=attempt)
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise Exception(f"Failed to fetch data for {ticker} after {self.max_retries} retries.")
        
        raise Exception(f"Failed to fetch data for {ticker}. Status: {response.status_code}")
