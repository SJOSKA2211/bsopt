import asyncio
import time

import structlog
from prometheus_client import Counter, Histogram

from src.api.providers import PolygonProvider, YahooProvider
from src.scrapers.engine import NSEScraper

logger = structlog.get_logger()

# 📊 METRICS: Track Data Mesh performance
ROUTING_COUNT = Counter(
    "market_data_routing_total",
    "Total count of market data requests",
    ["target", "market"],
)
ROUTING_LATENCY = Histogram(
    "market_data_routing_latency_seconds", "Latency of market data requests", ["target"]
)
SCRAPER_PARSE_SUCCESS = Counter(
    "market_data_scraper_success_total", "Success count of HTML parsing", ["market"]
)


class MarketDataRouter:
    """
    OPTIMIZED: Adaptive, latency-aware data routing engine.
    Uses EWMA to track provider performance and selects the optimal path.
    """

    def __init__(self):
        self.nse = NSEScraper()
        self.polygon = PolygonProvider()
        self.yahoo = YahooProvider()

        # Latency state (EWMA)
        # Higher score = more latency. Initialize with baseline estimates.
        self._latency_map = {
            "NSE": 0.5,  # High latency (Scraping)
            "Polygon": 0.05,  # Low latency (Direct API)
            "Yahoo": 0.1,  # Medium latency (Public API)
        }
        self._alpha = 0.2  # Smoothing factor for EWMA

    async def get_live_quote(self, symbol: str, market: str = "AUTO") -> dict:
        """Adaptive routing entry point."""
        start_time = time.time()

        # 1. Select candidates based on market
        candidates = []
        if market == "NSE" or symbol.endswith(".NR"):
            candidates = ["NSE", "Yahoo"]
        elif "-" in symbol and ("USD" in symbol or "USDT" in symbol):
            # Placeholder for crypto
            candidates = ["Yahoo"]
        else:
            candidates = ["Polygon", "Yahoo"]

        # 2. Sort by current EWMA latency (fastest first)
        sorted_candidates = sorted(candidates, key=lambda x: self._latency_map[x])

        last_error = None
        for provider_name in sorted_candidates:
            try:
                # Attempt fastest candidate
                provider_start = time.time()

                if provider_name == "NSE":
                    # Use get_ticker_data for NSE as it's the established method in the engine
                    res = await self.nse.get_ticker_data(symbol.replace(".NR", ""))
                    if "error" not in res:
                        SCRAPER_PARSE_SUCCESS.labels(market="NSE").inc()
                elif provider_name == "Polygon":
                    res = await self.polygon.get_ticker_data(symbol)
                else:
                    res = await self.yahoo.get_ticker_data(symbol)

                # Success: Update EWMA latency
                prov_latency = time.time() - provider_start
                self._latency_map[provider_name] = (
                    self._alpha * prov_latency
                    + (1 - self._alpha) * self._latency_map[provider_name]
                )

                total_latency = time.time() - start_time
                ROUTING_LATENCY.labels(target=provider_name).observe(total_latency)
                ROUTING_COUNT.labels(target=provider_name, market=market).inc()

                return res

            except Exception as e:
                last_error = e
                # Penalty for failure: Double the tracked latency
                self._latency_map[provider_name] *= 2.0
                logger.warning(
                    "provider_failed_failing_over", provider=provider_name, error=str(e)
                )

        raise last_error or Exception(f"No providers available for {symbol}")

    async def search_markets(self, query: str) -> list:
        """Global symbol search (Tickers + Metadata) - PARALLELISED."""
        results = []
        try:
            # OPTIMIZED: Run searches concurrently
            poly_task = self.polygon.search(query)
            yahoo_task = (
                self.yahoo.yahoo_search(query)
                if hasattr(self.yahoo, "yahoo_search")
                else self.yahoo.search(query)
            )

            combined = await asyncio.gather(
                poly_task, yahoo_task, return_exceptions=True
            )

            for res in combined:
                if isinstance(res, list):
                    results.extend(res)
                elif isinstance(res, Exception):
                    logger.warning(
                        "provider_search_partial_failure", error=str(res)
                    )

            logger.info(
                "market_search_completed", query=query, results_count=len(results)
            )
        except Exception as e:
            logger.error("market_search_failed", query=query, error=str(e))
        return results

    async def get_option_chain_snapshot(self, symbol: str) -> list:
        """Fetch a full option chain snapshot - RACING PATTERN."""
        try:
            # OPTIMIZED: Race providers for the fastest valid response
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
