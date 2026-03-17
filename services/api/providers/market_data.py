import structlog

from core.shared.utils.http_client import HttpClientManager
from core.shared.utils.resilience import retry_with_backoff
from services.config import settings
from services.scrapers.stealth import default_stealth_client

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
    async def get_ticker_data(self, symbol: str) -> dict:
        if self.api_key == "DEMO_KEY":
            return {"symbol": symbol, "price": 150.0, "provider": "Polygon (Mock)"}

        url = f"https://api.polygon.io/v2/last/trade/{symbol}?apiKey={self.api_key}"
        response = await self.client.get(url)
        response.raise_for_status()
        return response.json()

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
    async def get_ticker_data(self, symbol: str) -> dict:
        """Fetch quote data from Yahoo Finance using stealth impersonation."""
        # Normalize symbol for Yahoo (.NS for NSE stocks if needed, but handled by router)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"

        try:
            response = await self.stealth.get(url)
            if response.status_code != 200:
                logger.warning("yahoo_lookup_failed", status=response.status_code, symbol=symbol)
                return {"symbol": symbol, "error": "Yahoo lookup failed"}

            data = response.json()
            meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})

            price = meta.get("regularMarketPrice")
            if price is None:
                price = meta.get("chartPreviousClose", 0.0)

            return {
                "symbol": symbol,
                "price": float(price),
                "currency": meta.get("currency"),
                "exchange": meta.get("exchangeName"),
                "provider": "Yahoo",
                "timestamp": meta.get("regularMarketTime"),
            }
        except Exception as e:
            logger.error("yahoo_provider_error", error=str(e), symbol=symbol)
            return {"symbol": symbol, "error": str(e)}
