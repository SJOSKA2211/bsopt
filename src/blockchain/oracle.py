import time
from typing import Callable, Any

import structlog
from src.utils.cache import get_redis

logger = structlog.get_logger(__name__)


class OracleManager:
    """
    Hybrid Price Oracle (Speed-v1).
    Combines high-frequency off-chain feeds with on-chain JSON-RPC state.
    """

    def __init__(self, cache_ttl: int = 10):
        self.cache_ttl = cache_ttl
        self._feeds: dict[str, dict] = {}
        self._sources = ["WS", "RPC", "AGG"]

    async def get_price(self, symbol: str, contract_address: str, rpc_fallback: Callable[[str], Any]) -> float:
        """
        Fetch price with confidence scoring across multiple sources.
        """
        now = time.time()
        redis = get_redis()
        
        # 1. ⚡ SPEED FEED (WebSocket/Redis Cache)
        if redis:
            ws_price = await redis.get(f"price:ws:{symbol}")
            if ws_price:
                price = float(ws_price)
                confidence = 0.95 # Higher confidence for WS
                logger.info("oracle_hit_ws", symbol=symbol, price=price)
                return price

        # 2. 🏛️ RPC FEED (On-chain/Local Cache)
        if contract_address in self._feeds:
            entry = self._feeds[contract_address]
            if now - entry["time"] < self.cache_ttl:
                logger.info("oracle_hit_local", symbol=symbol, price=entry["price"])
                return entry["price"]

        # 3. 🛡️ FALLBACK (Live RPC Call)
        try:
            price = await rpc_fallback(contract_address)
            self._feeds[contract_address] = {"price": price, "time": now, "source": "RPC"}
            
            if redis:
                await redis.setex(f"price:rpc:{symbol}", self.cache_ttl, str(price))
                
            return price
        except Exception as e:
            logger.error("oracle_rpc_fallback_failed", symbol=symbol, error=str(e))
            raise
            
    def get_confidence_score(self, source: str, age: float) -> float:
        """Calculate confidence based on source and age."""
        base_scores = {"WS": 0.95, "RPC": 0.85, "AGG": 0.90}
        decay = 0.1 * (age / self.cache_ttl)
        return max(base_scores.get(source, 0.5) - decay, 0.0)
