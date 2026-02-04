from typing import Dict, List
from src.utils.resilience import retry_with_backoff

class PolygonProvider:
    @retry_with_backoff(retries=3, initial_delay=1.0)
    async def get_ticker_data(self, symbol: str) -> Dict:
        return {"symbol": symbol, "price": 150.0, "provider": "Polygon"}

    async def search(self, query: str) -> List[Dict]:
        return [{"symbol": query, "name": f"{query} Corp", "provider": "Polygon"}]

    @retry_with_backoff(retries=2, initial_delay=2.0)
    async def get_option_chain(self, symbol: str) -> List[Dict]:
        return [{"symbol": symbol, "strike": 100.0, "expiry": "2026-12-19", "type": "call", "price": 5.0}]

class YahooProvider:
    @retry_with_backoff(retries=3, initial_delay=1.0)
    async def get_ticker_data(self, symbol: str) -> Dict:
        return {"symbol": symbol, "price": 149.5, "provider": "Yahoo"}

    async def search(self, query: str) -> List[Dict]:
        return [{"symbol": query, "name": f"{query} Inc", "provider": "Yahoo"}]

    @retry_with_backoff(retries=2, initial_delay=2.0)
    async def get_option_chain(self, symbol: str) -> List[Dict]:
        return [{"symbol": symbol, "strike": 100.0, "expiry": "2026-12-19", "type": "put", "price": 4.5}]
