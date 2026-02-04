import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.scrapers.engine import NSEScraper

@pytest.mark.asyncio
async def test_tab_multiplexing_concurrency():
    # Setup Mocks correctly
    # row is returned by page.locator() which is a synchronous call
    mock_row = MagicMock()
    mock_row.count = AsyncMock(return_value=1)
    
    # price_cell is returned by row.locator() which is also synchronous
    mock_price_cell = MagicMock()
    mock_price_cell.inner_text = AsyncMock(return_value="10.0")
    
    # Configure row.locator to return different things if needed, or just mock it
    mock_row.locator.return_value = mock_price_cell
    
    # page.locator is synchronous
    mock_page = AsyncMock() # dispatch etc are async
    mock_page.locator = MagicMock(return_value=mock_row)
    mock_page.goto = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.close = AsyncMock()
    
    # browser.new_page is a coroutine
    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    
    mock_playwright_instance = AsyncMock()
    mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    
    with patch("src.scrapers.engine.async_playwright") as mock_ap:
        # async_playwright() returns a context manager, but in my code I used .start()
        # self.playwright = await async_playwright().start()
        mock_ap.return_value.start = AsyncMock(return_value=mock_playwright_instance)
        
        scraper = NSEScraper()
        
        # Trigger multiple requests concurrently
        symbols = ["SCOM", "KCB", "EQTY", "EABL", "BAT"]
        tasks = [scraper.get_ticker_data(s) for s in symbols]
        
        results = await asyncio.gather(*tasks)
        
        # Verify browser was only launched once
        assert mock_playwright_instance.chromium.launch.call_count == 1
        
        # Verify new_page was called for each symbol
        assert mock_browser.new_page.call_count == 5
        
        for r in results:
            if "error" in r:
                pytest.fail(f"Scrape failed with error: {r['error']}")
            assert "price" in r
            assert r["price"] == 10.0

@pytest.mark.asyncio
async def test_scraper_shutdown():
    scraper = NSEScraper()
    scraper.browser = AsyncMock()
    scraper.playwright = AsyncMock()
    
    await scraper.shutdown()
    assert scraper.browser.close.called
    assert scraper.playwright.stop.called
