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
    
    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io", max_retries: int = 3, provider: str = "auto"):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.provider = provider
        if provider == "auto":
             if "polygon" in base_url:
                 self.provider = "polygon"
             else:
                 self.provider = "alpha_vantage"
        self.api_name = self.provider

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
        # MOCK MODE FOR DEMO/TESTING
        # Use mock if key is DEMO_KEY, empty, or None, or provider is mock.
        if not self.api_key or self.api_key.strip() == "DEMO_KEY" or self.provider == "mock":
            logger.info("scrape_mock_mode", ticker=ticker, reason="Using DEMO_KEY or mock provider")
            # Generate mock data using pandas date_range
            dates = pd.date_range(start=start_date, end=end_date, freq="B") # Business days
            # Create synthetic price movement
            base_price = 150.0
            data = []
            import numpy as np
            
            for i, _ in enumerate(dates):
                # Simple random walk
                change = np.random.uniform(-2, 2)
                base_price += change
                
                row = {
                    "timestamp": int(dates[i].timestamp() * 1000), # Polygon uses millis
                    "open": base_price,
                    "high": base_price + np.random.uniform(0, 5),
                    "low": base_price - np.random.uniform(0, 5),
                    "close": base_price + np.random.uniform(-1, 1),
                    "volume": int(np.random.uniform(100000, 5000000))
                }
                data.append(row)
                
            df = pd.DataFrame(data)
            
            duration = 0.1
            SCRAPE_DURATION.labels(api="mock").observe(duration)
            logger.info("scrape_success", ticker=ticker, duration=duration, rows=len(df), mode="mock")
            
            return df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Determine API provider based on self.provider set in __init__
        
        # Alpha Vantage Implementation
        if self.provider == "alpha_vantage":
             url = "https://www.alphavantage.co/query"
             params = {
                 "function": "TIME_SERIES_DAILY",
                 "symbol": ticker,
                 "apikey": self.api_key,
                 "outputsize": "full",
                 "datatype": "json"
             }
             
             start_time = time.time()
             for attempt in range(self.max_retries + 1):
                try:
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        if "Time Series (Daily)" in data:
                            ts_data = data["Time Series (Daily)"]
                            records = []
                            for date_str, values in ts_data.items():
                                # Filter by date range
                                if start_date <= date_str <= end_date:
                                    records.append({
                                        "timestamp": pd.Timestamp(date_str).value // 10**6, # ms
                                        "open": float(values["1. open"]),
                                        "high": float(values["2. high"]),
                                        "low": float(values["3. low"]),
                                        "close": float(values["4. close"]),
                                        "volume": int(values["5. volume"])
                                    })
                            
                            df = pd.DataFrame(records)
                            if not df.empty:
                                df = df.sort_values("timestamp")
                                duration = time.time() - start_time
                                SCRAPE_DURATION.labels(api="alpha_vantage").observe(duration)
                                logger.info("scrape_success", ticker=ticker, duration=duration, attempt=attempt, api="alpha_vantage")
                                return df[["timestamp", "open", "high", "low", "close", "volume"]]
                            else:
                                logger.warning("scrape_empty", ticker=ticker, reason="No data in range")
                                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
                        
                        elif "Error Message" in data:
                             logger.error("scrape_error", ticker=ticker, error=data["Error Message"])
                             raise Exception(f"Alpha Vantage Error: {data['Error Message']}")
                        elif "Note" in data: # Rate limit
                             logger.warning("scrape_rate_limit", ticker=ticker, note=data["Note"])
                             time.sleep(60) # Wait for rate limit reset
                             continue
                    
                    elif response.status_code == 401:
                         SCRAPE_ERRORS.labels(api="alpha_vantage", status_code=401).inc()
                         raise Exception("401 Unauthorized: Invalid Alpha Vantage API Key.")
                    else:
                        SCRAPE_ERRORS.labels(api="alpha_vantage", status_code=response.status_code).inc()
                        logger.warning("scrape_http_error", ticker=ticker, status_code=response.status_code, attempt=attempt)
                    
                    if attempt < self.max_retries:
                        time.sleep(1)
                        continue

                except Exception as e:
                    SCRAPE_ERRORS.labels(api="alpha_vantage", status_code="exception").inc()
                    logger.error("scrape_exception", ticker=ticker, error=str(e))
                    if attempt < self.max_retries:
                        continue
                    raise

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
                elif response.status_code == 401:
                    logger.error("scrape_unauthorized", ticker=ticker, message="Invalid API Key. Please check POLYGON_API_KEY.", attempt=attempt)
                    SCRAPE_ERRORS.labels(api=self.api_name, status_code=response.status_code).inc()
                    # Do not retry on 401
                    raise Exception("401 Unauthorized: Invalid API Key.")
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
