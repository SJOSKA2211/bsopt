import httpx
import pandas as pd
import time
import asyncio
import structlog
import re
from src.shared.observability import SCRAPE_DURATION, SCRAPE_ERRORS
from src.utils.http_client import HttpClientManager

logger = structlog.get_logger()

class MarketDataScraper:
    def __init__(self, api_key: str, provider: str = "alpha_vantage", max_retries: int = 3):
        self.api_key = api_key
        self.provider = provider
        self.max_retries = max_retries
        self.base_url = "https://www.alphavantage.co/query" if provider == "alpha_vantage" else "https://api.polygon.io"

    def _validate_inputs(self, ticker: str, start_date: str, end_date: str):
        """Validates ticker and date formats to prevent injection/traversal."""
        if not re.match(r"^[A-Z0-9.-]{1,20}$", ticker):
            raise ValueError(f"Invalid ticker symbol: {ticker}")
        date_pattern = r"^\d{4}-\d{2}-\d{2}$"
        if not re.match(date_pattern, start_date) or not re.match(date_pattern, end_date):
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

    def _redact_message(self, message: str) -> str:
        """Redacts the API key from strings (e.g., URLs in exceptions)."""
        if not self.api_key or self.api_key == "DEMO_KEY":
            return message
        return message.replace(self.api_key, "[REDACTED]")

    async def fetch_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical daily data for a given ticker and date range asynchronously."""
        self._validate_inputs(ticker, start_date, end_date)
        last_response = None
        client = HttpClientManager.get_client()
        
        # Alpha Vantage Implementation
        if self.provider == "alpha_vantage" or self.provider == "auto":
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": "full",
                "apikey": self.api_key
            }
            
            start_time = time.time()
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.get(self.base_url, params=params)
                    last_response = response
                    if response.status_code == 200:
                        data = response.json()
                        if "Time Series (Daily)" in data:
                            time_series = data["Time Series (Daily)"]
                            records = []
                            for date_str, values in time_series.items():
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
                                if self.provider == "auto":
                                    break
                                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
                        elif "Error Message" in data:
                            logger.error("scrape_error", ticker=ticker, error=data["Error Message"])
                            if self.provider == "auto":
                                break
                            raise Exception(f"Alpha Vantage Error: {data['Error Message']}")
                        elif "Note" in data: # Rate limit
                            logger.warning("scrape_rate_limit", ticker=ticker, note=data["Note"])
                            if attempt < self.max_retries:
                                await asyncio.sleep(0.1)
                                continue
                            else:
                                if self.provider == "auto":
                                    break
                                raise Exception("Alpha Vantage rate limit reached.")
                    elif response.status_code == 401:
                        SCRAPE_ERRORS.labels(api="alpha_vantage", status_code=401).inc()
                        if self.provider == "auto":
                            break
                        raise Exception("401 Unauthorized: Invalid Alpha Vantage API Key.")
                    else:
                        SCRAPE_ERRORS.labels(api="alpha_vantage", status_code=response.status_code).inc()
                        logger.warning("scrape_http_error", ticker=ticker, status_code=response.status_code, attempt=attempt)
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.1)
                        continue

                except Exception as e:
                    SCRAPE_ERRORS.labels(api="alpha_vantage", status_code="exception").inc()
                    logger.error("scrape_exception", ticker=ticker, error=self._redact_message(str(e)))
                    if attempt < self.max_retries:
                        continue
                    if self.provider == "auto":
                        break
                    raise

        # Polygon.io Implementation
        if self.provider == "polygon" or self.provider == "auto":
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            params = {"apiKey": self.api_key}
            
            start_time = time.time()
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.get(url, params=params)
                    last_response = response
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "OK" and "results" in data:
                            df = pd.DataFrame(data["results"])
                            df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                            duration = time.time() - start_time
                            SCRAPE_DURATION.labels(api="polygon").observe(duration)
                            logger.info("scrape_success", ticker=ticker, duration=duration, attempt=attempt, api="polygon")
                            return df[["timestamp", "open", "high", "low", "close", "volume"]]
                        else:
                            logger.warning("scrape_api_error", ticker=ticker, status=data.get("status"), attempt=attempt)
                    elif response.status_code == 401:
                        SCRAPE_ERRORS.labels(api="polygon", status_code=401).inc()
                        raise Exception("401 Unauthorized: Invalid API Key.")
                    else:
                        SCRAPE_ERRORS.labels(api="polygon", status_code=response.status_code).inc()
                        logger.warning("scrape_http_error", ticker=ticker, status_code=response.status_code, attempt=attempt)
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.1)
                        continue
                    
                except httpx.RequestError as e:
                    SCRAPE_ERRORS.labels(api="polygon", status_code="exception").inc()
                    logger.error("scrape_exception", ticker=ticker, error=self._redact_message(str(e)), attempt=attempt)
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.1)
                        continue
                    raise Exception(f"Failed to fetch data for {ticker} after {self.max_retries} retries.")
        
        status_code = last_response.status_code if last_response else "No response"
        raise Exception(f"Failed to fetch data for {ticker}. Status: {status_code}")

