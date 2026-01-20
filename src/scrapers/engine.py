import asyncio
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
    Uses Tab Multiplexing to handle concurrent requests efficiently.
    """
    BASE_URL = "https://www.nse.co.ke/market-statistics/equity-statistics"

    def __init__(self):
        self.playwright = None
        self.browser = None
        self._lock = asyncio.Lock()

    async def _ensure_browser(self):
        async with self._lock:
            if not self.browser:
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(args=["--no-sandbox"])

    async def get_ticker_data(self, symbol: str) -> Dict:
        await self._ensure_browser()
        
        # Create a new page (tab) for this request
        page = await self.browser.new_page()
        try:
            # Navigate to NSE live stats
            await page.goto(self.BASE_URL, timeout=30000)
            # Wait for the data table to load
            await page.wait_for_selector("table.equity-statistics")
            
            # Extract row for specific symbol
            row = page.locator(f"//tr[td[contains(text(), '{symbol}')]]")
            if await row.count() == 0:
                logger.error("nse_ticker_not_found", symbol=symbol)
                return {"symbol": symbol, "error": "Ticker not found", "market": "NSE"}

            # Extract columns
            data = {
                "symbol": symbol,
                "market": "NSE",
                "price": await row.locator("td:nth-child(3)").inner_text(),
                "change": await row.locator("td:nth-child(4)").inner_text(),
                "volume": await row.locator("td:nth-child(8)").inner_text(),
                "timestamp": datetime.now().isoformat()
            }
            
            return self._clean_data(data)
        except Exception as e:
            logger.error("nse_scrape_failed", error=str(e), symbol=symbol)
            return {"symbol": symbol, "error": str(e), "market": "NSE"}
        finally:
            # Close the tab, but keep the browser running
            await page.close()

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