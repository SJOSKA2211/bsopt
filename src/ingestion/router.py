import asyncio
import time

import msgspec
import structlog

from api.providers import PolygonProvider, YahooProvider
from src.ingestion.engine import NSEScraper
from src.shared.schemas.market import MarketQuote

logger = structlog.get_logger()

# 📊 METRICS: Track Data Mesh performance
from src.shared.observability import (  # noqa: E402
    ROUTING_COUNT,
    ROUTING_LATENCY,
)


class MarketDataRouter:
    """
    OPTIMIZED: Adaptive, latency-aware data routing engine.
    Uses EWMA to track provider performance and selects the optimal path.
    Shared across instances via Redis for global coordination.
    """

    def __init__(self):
        self.nse = NSEScraper()
        self.polygon = PolygonProvider()
        self.yahoo = YahooProvider()
        from src.shared.utils.cache import get_redis

        self.redis = get_redis()

        # Latency state (EWMA)
        # Higher score = more latency. Initialize with baseline estimates.
        self._local_latency_map = {
            "NSE": 0.5,  # High latency (Scraping)
            "Polygon": 0.05,  # Low latency (Direct API)
            "Yahoo": 0.1,  # Medium latency (Public API)
        }
        self._alpha = 0.2  # Smoothing factor for EWMA

    async def _get_latency_map(self) -> dict[str, float]:
        """Fetch current latency scores, merging local and Redis global state."""
        if not self.redis:
            return self._local_latency_map

        try:
            global_latencies = await self.redis.hgetall("market_data:latency_map")
            if global_latencies:
                # Merge: Redis overrides local for faster convergence across nodes
                return {k.decode(): float(v) for k, v in global_latencies.items()}
        except Exception:
            pass
        return self._local_latency_map

    async def _update_latency(self, provider: str, latency: float):
        """Update EWMA score locally and in Redis."""
        # 1. Local update
        old_val = self._local_latency_map.get(provider, 0.1)
        new_val = self._alpha * latency + (1 - self._alpha) * old_val
        self._local_latency_map[provider] = new_val

        # 2. Redis update (Global visibility)
        if self.redis:
            try:
                await self.redis.hset("market_data:latency_map", provider, str(new_val))
            except Exception:
                pass

    async def get_live_quote(self, symbol: str, market: str = "AUTO") -> MarketQuote:
        """
        God-Tier: Speculative Concurrency Router.
        Races providers with a staggered start to ensure minimal latency.
        """
        start_time = time.time()
        latency_map = await self._get_latency_map()

        # 1. Select candidates
        candidates = []
        if market == "NSE" or symbol.endswith(".NR"):
            candidates = ["NSE", "Yahoo"]
        elif "-" in symbol and ("USD" in symbol or "USDT" in symbol):
            candidates = ["Yahoo"]
        else:
            candidates = ["Polygon", "Yahoo"]

        # 2. Sort by current EWMA latency
        sorted_candidates = sorted(candidates, key=lambda x: latency_map.get(x, 0.1))

        # 3.  SPECULATIVE RACE
        async def _call_provider(provider_name) -> tuple[MarketQuote, str]:
            try:
                if provider_name == "NSE":
                    res = await self.nse.get_ticker_data(symbol.replace(".NR", ""))
                elif provider_name == "Polygon":
                    res = await self.polygon.get_ticker_data(symbol)
                else:
                    res = await self.yahoo.get_ticker_data(symbol)

                # Update EWMA
                p_latency = time.time() - start_time
                await self._update_latency(provider_name, p_latency)

                # Cache successful result for high-availability fallback
                if self.redis:
                    await self.redis.setex(
                        f"market_cache:{symbol}",
                        3600,  # 1 hour cache
                        msgspec.json.encode(res),
                    )

                return res, provider_name
            except Exception as e:
                # Penalty for failure
                await self._update_latency(provider_name, latency_map.get(provider_name, 0.1) * 2.0)
                raise e

        # Staggered launch
        tasks = []
        for i, provider in enumerate(sorted_candidates):
            tasks.append(asyncio.create_task(_call_provider(provider)))
            if i < len(sorted_candidates) - 1:
                wait_time = min(0.2, latency_map.get(sorted_candidates[0], 0.1) * 0.5)
                done, _ = await asyncio.wait(
                    tasks, timeout=wait_time, return_when=asyncio.FIRST_COMPLETED
                )
                if done:
                    for t in done:
                        try:
                            res, p_name = t.result()
                            for remaining in tasks:
                                if not remaining.done():
                                    remaining.cancel()

                            total_latency = time.time() - start_time
                            ROUTING_LATENCY.labels(target=p_name).observe(total_latency)
                            ROUTING_COUNT.labels(target=p_name, market=market).inc()
                            return res
                        except Exception:
                            continue

        # 4. Final wait if no one finished during staggered launch
        if not tasks:
            return await self._get_fallback_quote(symbol)

        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                try:
                    res, p_name = t.result()
                    for p in pending:
                        p.cancel()

                    total_latency = time.time() - start_time
                    ROUTING_LATENCY.labels(target=p_name).observe(total_latency)
                    ROUTING_COUNT.labels(target=p_name, market=market).inc()
                    return res
                except Exception:
                    tasks.remove(t)

            if not tasks:
                break

        return await self._get_fallback_quote(symbol)

    async def _get_fallback_quote(self, symbol: str) -> MarketQuote:
        """
        High-Availability Cache Fallback: Retrieves the last known successful quote from Redis.
        Ensures the UI stays alive during transient provider outages using real historical data.
        """
        if self.redis:
            try:
                cached = await self.redis.get(f"market_cache:{symbol}")
                if cached:
                    logger.info("routing_fallback_to_cache_success", symbol=symbol)
                    return msgspec.json.decode(cached, type=MarketQuote)
            except Exception as e:
                logger.warning("routing_cache_fallback_failed", symbol=symbol, error=str(e))

        raise Exception(f"All providers and cache failed for {symbol}")

    async def search_markets(self, query: str) -> list:
        """Global symbol search (Tickers + Metadata) - PARALLELISED."""
        results = []
        try:
            poly_task = self.polygon.search(query)
            yahoo_task = (
                self.yahoo.yahoo_search(query)
                if hasattr(self.yahoo, "yahoo_search")
                else self.yahoo.search(query)
            )

            combined = await asyncio.gather(poly_task, yahoo_task, return_exceptions=True)

            for res in combined:
                if isinstance(res, list):
                    results.extend(res)
                elif isinstance(res, Exception):
                    logger.warning("provider_search_partial_failure", error=str(res))

            logger.info("market_search_completed", query=query, results_count=len(results))
        except Exception as e:
            logger.error("market_search_failed", query=query, error=str(e))
        return results

    async def get_option_chain_snapshot(self, symbol: str) -> list:
        """Fetch a full option chain snapshot - RACING PATTERN."""
        try:
            poly_task = asyncio.create_task(self.polygon.get_option_chain(symbol))
            yahoo_task = asyncio.create_task(self.yahoo.get_option_chain(symbol))

            # Simple racing: Wait for first to complete, but check if it's empty
            # A more complex version would wait for the first *non-empty* result
            done, pending = await asyncio.wait(
                [poly_task, yahoo_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                res = task.result()
                if res:
                    # Cancel pending tasks to save resources
                    for p in pending:
                        p.cancel()
                    logger.info(
                        "option_chain_race_winner",
                        symbol=symbol,
                        contracts_count=len(res),
                    )
                    return res

            # If the first was empty, wait for the rest
            if pending:
                done2, _ = await asyncio.wait(pending)
                for task in done2:
                    res = task.result()
                    if res:
                        return res

            return []
        except Exception as e:
            logger.error("option_chain_fetch_failed", symbol=symbol, error=str(e))
            return []
