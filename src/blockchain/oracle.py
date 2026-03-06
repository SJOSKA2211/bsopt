import time
from collections.abc import Callable
from typing import Any

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

    async def get_price(
        self, symbol: str, contract_address: str, rpc_fallback: Callable[[str], Any]
    ) -> float:
        """
        Fetch price with confidence scoring across multiple sources.
        """
        now = time.time()
        redis = get_redis()

        # 1. ⚡ SPEED FEED (WebSocket/Redis Cache)
        if redis:
            ws_price = await redis.get(f"price:ws:{symbol}")
            ws_ts = await redis.get(f"price:ws:{symbol}:ts")
            
            if ws_price and ws_ts:
                price = float(ws_price)
                age = now - float(ws_ts)
                confidence = self.get_confidence_score("WS", age)
                
                if confidence > 0.8:
                    logger.info("oracle_hit_ws", symbol=symbol, price=price, confidence=round(confidence, 2))
                    return price

        # 2. 🏛️ RPC FEED (On-chain/Local Cache)
        if contract_address in self._feeds:
            entry = self._feeds[contract_address]
            age = now - entry["time"]
            if age < self.cache_ttl:
                logger.info("oracle_hit_local", symbol=symbol, price=entry["price"], source=entry["source"])
                return entry["price"]

        # 3. 🛡️ FALLBACK (Live RPC Call)
        try:
            price = await rpc_fallback(contract_address)
            self._feeds[contract_address] = {"price": price, "time": now, "source": "RPC"}

            if redis:
                await redis.setex(f"price:rpc:{symbol}", self.cache_ttl, str(price))
                await redis.setex(f"price:rpc:{symbol}:ts", self.cache_ttl, str(now))

            return price
        except Exception as e:
            logger.error("oracle_rpc_fallback_failed", symbol=symbol, error=str(e))
            raise

    def get_confidence_score(self, source: str, age: float) -> float:
        """Calculate confidence based on source and age."""
        base_scores = {"WS": 0.95, "RPC": 0.85, "AGG": 0.90}
        decay = 0.1 * (age / self.cache_ttl)
        return max(base_scores.get(source, 0.5) - decay, 0.0)
