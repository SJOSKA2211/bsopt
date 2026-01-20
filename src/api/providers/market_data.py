from typing import Dict, List

class PolygonProvider:
    async def get_ticker_data(self, symbol: str) -> Dict:
        return {"symbol": symbol, "price": 150.0, "provider": "Polygon"}

    async def search(self, query: str) -> List[Dict]:
        return [{"symbol": query, "name": f"{query} Corp", "provider": "Polygon"}]

    async def get_option_chain(self, symbol: str) -> List[Dict]:
        return [{"symbol": symbol, "strike": 100.0, "expiry": "2026-12-19", "type": "call", "price": 5.0}]

class YahooProvider:
    async def get_ticker_data(self, symbol: str) -> Dict:
        return {"symbol": symbol, "price": 149.5, "provider": "Yahoo"}

    async def search(self, query: str) -> List[Dict]:
        return [{"symbol": query, "name": f"{query} Inc", "provider": "Yahoo"}]

    async def get_option_chain(self, symbol: str) -> List[Dict]:
        return [{"symbol": symbol, "strike": 100.0, "expiry": "2026-12-19", "type": "put", "price": 4.5}]
