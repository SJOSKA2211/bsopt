from src.scrapers.engine import NSEScraper
from src.api.providers import PolygonProvider, YahooProvider
import structlog
import time
from prometheus_client import Counter, Histogram

logger = structlog.get_logger()

# 📊 METRICS: Track Data Mesh performance
ROUTING_COUNT = Counter("market_data_routing_total", "Total count of market data requests", ["target", "market"])
ROUTING_LATENCY = Histogram("market_data_routing_latency_seconds", "Latency of market data requests", ["target"])
SCRAPER_PARSE_SUCCESS = Counter("market_data_scraper_success_total", "Success count of HTML parsing", ["market"])

class MarketDataRouter:
    """
    Intelligent routing logic to find data anywhere.
    """
    def __init__(self):
        self.nse = NSEScraper()
        self.polygon = PolygonProvider()
        self.yahoo = YahooProvider()

    async def get_live_quote(self, symbol: str, market: str = "AUTO") -> dict:
        """ Unified entry point for Frontend and ML engine. """
        start_time = time.time()
        
        # 1. Direct Routing based on Market Suffix
        if market == "NSE" or symbol.endswith(".NR"):
            ROUTING_COUNT.labels(target="NSE", market="NSE").inc()
            logger.info("routing_to_scraper", target="NSE", symbol=symbol)
            res = await self.nse.get_ticker_data(symbol.replace(".NR", ""))
            
            if "error" not in res:
                SCRAPER_PARSE_SUCCESS.labels(market="NSE").inc()
            
            ROUTING_LATENCY.labels(target="NSE").observe(time.time() - start_time)
            return res

        # 2. Crypto Routing
        if "-" in symbol and ("USD" in symbol or "USDT" in symbol):
            logger.info("routing_to_crypto", symbol=symbol)
            # Placeholder for CCXT logic

        # 3. Default High-Frequency API (US Market)
        try:
            ROUTING_COUNT.labels(target="Polygon", market="US").inc()
            res = await self.polygon.get_ticker_data(symbol)
            ROUTING_LATENCY.labels(target="Polygon").observe(time.time() - start_time)
            return res
        except Exception as e:
            # 4. Fallback to Yahoo
            ROUTING_COUNT.labels(target="Yahoo", market="Global").inc()
            logger.warn("fallback_to_yahoo", symbol=symbol, error=str(e))
            res = await self.yahoo.get_ticker_data(symbol)
            ROUTING_LATENCY.labels(target="Yahoo").observe(time.time() - start_time)
            return res

    async def search_markets(self, query: str) -> list:
        """ Global symbol search (Tickers + Metadata) """
        results = []
        try:
            # Aggregate results from multiple providers
            poly_results = await self.polygon.search(query)
            yahoo_results = await self.yahoo.yahoo_search(query) if hasattr(self.yahoo, 'yahoo_search') else await self.yahoo.search(query)
            results = poly_results + yahoo_results
            logger.info("market_search_completed", query=query, results_count=len(results))
        except Exception as e:
            logger.error("market_search_failed", query=query, error=str(e))
        return results

    async def get_option_chain_snapshot(self, symbol: str) -> list:
        """ Fetch a full option chain snapshot for calibration. """
        try:
            # Default to Polygon for high-fidelity option chains
            chain = await self.polygon.get_option_chain(symbol)
            if not chain:
                # Fallback to Yahoo
                chain = await self.yahoo.get_option_chain(symbol)
            
            logger.info("option_chain_fetched", symbol=symbol, contracts_count=len(chain))
            return chain
        except Exception as e:
            logger.error("option_chain_fetch_failed", symbol=symbol, error=str(e))
            return []
