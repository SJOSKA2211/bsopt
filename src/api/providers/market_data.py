
from src.utils.resilience import retry_with_backoff


class PolygonProvider:
    @retry_with_backoff(retries=3, initial_delay=1.0)
    async def get_ticker_data(self, symbol: str) -> dict:
        return {"symbol": symbol, "price": 150.0, "provider": "Polygon"}

    async def search(self, query: str) -> list[dict]:
        return [{"symbol": query, "name": f"{query} Corp", "provider": "Polygon"}]

    @retry_with_backoff(retries=2, initial_delay=2.0)
    async def get_option_chain(self, symbol: str) -> list[dict]:
        return [{"symbol": symbol, "strike": 100.0, "expiry": "2026-12-19", "type": "call", "price": 5.0}]

class YahooProvider:
    @retry_with_backoff(retries=3, initial_delay=1.0)
    async def get_ticker_data(self, symbol: str) -> dict:
        return {"symbol": symbol, "price": 149.5, "provider": "Yahoo"}

    async def search(self, query: str) -> list[dict]:
        return [{"symbol": query, "name": f"{query} Inc", "provider": "Yahoo"}]

    @retry_with_backoff(retries=2, initial_delay=2.0)
    async def get_option_chain(self, symbol: str) -> list[dict]:
        return [{"symbol": symbol, "strike": 100.0, "expiry": "2026-12-19", "type": "put", "price": 4.5}]
