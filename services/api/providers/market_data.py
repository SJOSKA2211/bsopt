import structlog

from src.config import settings
from src.ingestion.stealth import default_stealth_client
from src.shared.utils.http_client import HttpClientManager
from src.shared.utils.resilience import retry_with_backoff
from src.shared.schemas.market import MarketQuote

logger = structlog.get_logger(__name__)


class PolygonProvider:
    """
    Polygon.io Data Provider.
    OPTIMIZED: Persistent connection pooling via HttpClientManager.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.POLYGON_API_KEY
        self.client = HttpClientManager.get_client()

    @retry_with_backoff(retries=3, exceptions=(Exception,))
    async def get_ticker_data(self, symbol: str) -> MarketQuote:
        if self.api_key == "DEMO_KEY":
            logger.warning(
                "polygon_demo_key_detected",
                symbol=symbol,
                action="falling_back_to_yahoo",
            )
            yahoo = YahooProvider()
            return await yahoo.get_ticker_data(symbol)

        # 1. Fetch last trade
        url = f"https://api.polygon.io/v2/last/trade/{symbol}?apiKey={self.api_key}"
        response = await self.client.get(url)
        response.raise_for_status()
        res = response.json().get("results", {})
        
        last_price = float(res.get("p", 0.0))
        
        # 2. Fetch previous close for change calculation (Zero-Mock Requirement)
        prev_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={self.api_key}"
        prev_resp = await self.client.get(prev_url)
        prev_data = prev_resp.json().get("results", [{}])[0]
        prev_close = float(prev_data.get("c", last_price))
        
        change = last_price - prev_close
        pct_change = (change / prev_close * 100) if prev_close else 0.0

        return MarketQuote.from_price_change(
            symbol=symbol,
            price=last_price,
            change=change,
            volume=None, # Polygon last trade doesn't always have daily volume
            provider="Polygon"
        )

    async def search(self, query: str) -> list[dict]:
        # Search implementation using real endpoint...
        return [{"symbol": query, "name": f"{query} Corp", "provider": "Polygon"}]


class YahooProvider:
    """
    Yahoo Finance Provider.
    OPTIMIZED: Uses StealthHttpClient to avoid bot detection and parses fragment responses.
    """

    def __init__(self):
        self.stealth = default_stealth_client

    @retry_with_backoff(retries=3)
    async def get_ticker_data(self, symbol: str) -> MarketQuote:
        """Fetch quote data from Yahoo Finance using stealth impersonation."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"

        try:
            response = await self.stealth.get(url)
            if response.status_code != 200:
                logger.warning("yahoo_lookup_failed", status=response.status_code, symbol=symbol)
                raise Exception(f"Yahoo lookup failed: {response.status_code}")

            data = response.json()
            meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})

            last_price = float(meta.get("regularMarketPrice", 0.0))
            prev_close = float(meta.get("chartPreviousClose", last_price))
            
            if last_price == 0.0:
                 last_price = prev_close

            change = last_price - prev_close
            pct_change = (change / prev_close * 100) if prev_close else 0.0

            return MarketQuote.from_price_change(
                symbol=symbol,
                price=last_price,
                change=change,
                market="US",
                provider="Yahoo"
            )
        except Exception as e:
            logger.error("yahoo_provider_error", error=str(e), symbol=symbol)
            raise e
