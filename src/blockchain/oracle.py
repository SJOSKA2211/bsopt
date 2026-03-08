import time
from collections.abc import Callable
from typing import Any

import structlog

from src.shared.shm_mesh import SharedMemoryRingBuffer
from src.utils.cache import get_redis

logger = structlog.get_logger(__name__)


class OracleManager:
    """
    Hybrid Price Oracle (Speed-v1).
    OPTIMIZED: Multilayered lookup (SHM -> Redis -> RPC) with confidence weighting.
    """

    def __init__(self, cache_ttl: int = 10):
        self.cache_ttl = cache_ttl
        self._feeds: dict[str, dict] = {}
        self._sources = ["SHM", "WS", "RPC"]
        self._mesh = None
        self._last_mesh_head = 0

    def _get_mesh(self):
        if self._mesh is None:
            try:
                self._mesh = SharedMemoryRingBuffer(create=False)
            except Exception:
                pass
        return self._mesh

    async def get_price(
        self, 
        symbol: str, 
        contract_address: str, 
        rpc_fallbacks: list[Callable[[str], Any]]
    ) -> float:
        """
        Fetch price with multilayered fallback and confidence scoring.
        """
        now = time.time()
        now_ns = time.time_ns()

        # 1. 🚀 SHM MESH (Ultra-Low Latency)
        mesh = self._get_mesh()
        if mesh:
            # Note: In a real implementation, we might maintain a local symbol map
            # updated by a background task reading the mesh.
            # For this revamp, we assume the mesh contains the latest ticks.
            ticks, new_head = mesh.read_latest_msgspec(self._last_mesh_head)
            if ticks:
                self._last_mesh_head = new_head
                # Update local cache with latest mesh ticks
                for tick in ticks:
                    self._feeds[tick.symbol] = {
                        "price": tick.price,
                        "time": tick.receive_ts_ns / 1e9,
                        "source": "SHM"
                    }

            if symbol in self._feeds and self._feeds[symbol]["source"] == "SHM":
                entry = self._feeds[symbol]
                age = now - entry["time"]
                if age < 1.0: # Very strict TTL for SHM
                    return entry["price"]

        # 2. ⚡ SPEED FEED (Redis Cache / WebSocket)
        redis = get_redis()
        if redis:
            keys = [f"price:ws:{symbol}", f"price:ws:{symbol}:ts"]
            values = await redis.mget(keys)
            
            if values[0] and values[1]:
                price = float(values[0])
                age = now - float(values[1])
                confidence = self.get_confidence_score("WS", age)
                
                if confidence > 0.9:
                    return price

        # 3. 🏛️ RPC FEED (On-chain/Local Cache)
        if contract_address in self._feeds:
            entry = self._feeds[contract_address]
            age = now - entry["time"]
            if age < self.cache_ttl:
                return entry["price"]

        # 4. 🛡️ MULTI-RPC AGGREGATION
        prices = []
        for rpc_call in rpc_fallbacks:
            try:
                price = await rpc_call(contract_address)
                prices.append(price)
            except Exception:
                continue

        if not prices:
            raise Exception(f"All sources failed for {symbol}")

        prices.sort()
        mid = len(prices) // 2
        median_price = (prices[mid] + prices[~mid]) / 2 if prices else 0.0
        
        self._feeds[contract_address] = {"price": median_price, "time": now, "source": "RPC"}
        return median_price

        def get_confidence_score(self, source: str, age: float) -> float:

