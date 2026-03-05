import asyncio
import structlog
from src.utils.cache import get_redis
from src.scrapers.mesh_publisher import get_market_publisher

logger = structlog.get_logger(__name__)

class OracleManager:
    """
    Hybrid Price Oracle (Speed-v1).
    Combines on-chain data with MarketMesh high-frequency off-chain feeds.
    """
    def __init__(self, cache_ttl: int = 5):
        self.cache_ttl = cache_ttl
        self._price_overrides = {} # Symbol -> Price

    async def get_price(self, symbol: str, contract_addr: str, w3_func) -> float:
        """
        Get price with layered fallbacks:
        1. MarketMesh (Shared Memory)
        2. Local Override (WebSocket)
        3. Redis Cache
        4. On-chain (Slowest)
        """
        # 1. MarketMesh check (Mocked logic for PRD integration)
        # In real impl, we'd read from SHMManager("market_mesh")
        
        # 2. Local Override
        if symbol in self._price_overrides:
            return self._price_overrides[symbol]

        # 3. Redis Cache
        redis = get_redis()
        if redis:
            cached = await redis.get(f"oracle_price:{symbol}")
            if cached:
                return float(cached)

        # 4. On-chain Fallback
        price = await w3_func(contract_addr)
        
        # Backfill cache
        if redis:
            await redis.setex(f"oracle_price:{symbol}", self.cache_ttl, str(price))
        
        return price

    def update_price_feed(self, symbol: str, price: float):
        """Update from WebSocket thread."""
        self._price_overrides[symbol] = price
        logger.debug("oracle_feed_updated", symbol=symbol, price=price)
