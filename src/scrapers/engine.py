import asyncio
import time
from typing import Dict, Optional, Protocol
from playwright.async_api import async_playwright
import structlog
from datetime import datetime

logger = structlog.get_logger()

class MarketSource(Protocol):
    async def get_ticker_data(self, symbol: str) -> Dict:
        ...

class NSEScraper:
    """
    Dedicated scraper for Nairobi Securities Exchange (NSE).
    Bypasses standard API limitations to get frontier market data.
    Uses Tab Multiplexing and In-Memory Caching for maximum efficiency.
    """
    BASE_URL = "https://www.nse.co.ke/market-statistics/equity-statistics"

    def __init__(self):
        self.playwright = None
        self.browser = None
        self._lock = asyncio.Lock()
        self._data_cache = {}
        self._last_refresh = 0
        self._cache_ttl = 60 # 1 minute cache

    async def _ensure_browser(self):
        async with self._lock:
            if not self.browser:
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox"])

    async def _refresh_cache(self):
        """Fetches all equity statistics and updates the cache."""
        async with self._lock:
            # Double-check if another task refreshed it while we were waiting for the lock
            if time.time() - self._last_refresh < self._cache_ttl:
                return

            await self._ensure_browser()
            page = await self.browser.new_page()
            try:
                logger.info("nse_refreshing_cache", url=self.BASE_URL)
                await page.goto(self.BASE_URL, timeout=60000, wait_until="networkidle")
                await page.wait_for_selector("table.equity-statistics", timeout=10000)
                
                # Extract all rows at once using JS for speed
                rows_data = await page.evaluate("""() => {
                    const rows = Array.from(document.querySelectorAll('table.equity-statistics tr')).slice(1);
                    return rows.map(row => {
                        const cols = row.querySelectorAll('td');
                        if (cols.length < 8) return null;
                        return {
                            symbol: cols[0].innerText.trim(),
                            price: cols[2].innerText.trim(),
                            change: cols[3].innerText.trim(),
                            volume: cols[7].innerText.trim()
                        };
                    }).filter(x => x !== null);
                }""")
                
                new_cache = {}
                timestamp = datetime.now().isoformat()
                for item in rows_data:
                    symbol = item['symbol']
                    new_cache[symbol] = self._clean_data({
                        "symbol": symbol,
                        "market": "NSE",
                        "price": item['price'],
                        "change": item['change'],
                        "volume": item['volume'],
                        "timestamp": timestamp
                    })
                
                self._data_cache = new_cache
                self._last_refresh = time.time()
                logger.info("nse_cache_updated", count=len(new_cache))
            except Exception as e:
                logger.error("nse_refresh_failed", error=str(e))
                raise e
            finally:
                await page.close()

    async def get_ticker_data(self, symbol: str) -> Dict:
        """Get ticker data, refreshing cache if necessary."""
        if time.time() - self._last_refresh > self._cache_ttl:
            try:
                await self._refresh_cache()
            except Exception:
                # If refresh fails, try to serve from stale cache if possible
                if not self._data_cache:
                    return {"symbol": symbol, "error": "Scraper unavailable", "market": "NSE"}

        data = self._data_cache.get(symbol)
        if not data:
            logger.warning("nse_ticker_not_in_cache", symbol=symbol)
            return {"symbol": symbol, "error": "Ticker not found", "market": "NSE"}
        
        return data

    async def shutdown(self):
        """Cleanly close the browser instance."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    def _clean_data(self, data: Dict) -> Dict:
        """Helper to convert string price/volume to numeric types."""
        try:
            if 'price' in data and isinstance(data['price'], str):
                data['price'] = float(data['price'].replace(',', ''))
            if 'volume' in data and isinstance(data['volume'], str):
                data['volume'] = int(data['volume'].replace(',', ''))
            return data
        except (ValueError, AttributeError):
            return data