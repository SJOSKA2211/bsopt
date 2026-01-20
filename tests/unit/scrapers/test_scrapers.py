import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.scrapers.engine import NSEScraper

@pytest.mark.asyncio
async def test_nse_scraper_stub():
    # Mocking browser to avoid needing playwright install in unit tests
    mock_row = MagicMock()
    mock_row.count = AsyncMock(return_value=1)
    
    mock_price_cell = MagicMock()
    mock_price_cell.inner_text = AsyncMock(return_value="10.0")
    mock_row.locator.return_value = mock_price_cell
    
    mock_page = AsyncMock()
    # Explicitly make locator a MagicMock so it doesn't return a coroutine
    mock_page.locator = MagicMock(return_value=mock_row)
    
    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    
    mock_playwright_instance = AsyncMock()
    mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    
    with patch("src.scrapers.engine.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=mock_playwright_instance)
        
        scraper = NSEScraper()
        result = await scraper.get_ticker_data("SCOM")
        
        if "error" in result:
            pytest.fail(f"Scrape failed with error: {result['error']}")
            
        assert result["symbol"] == "SCOM"
        assert result["market"] == "NSE"
        assert result["price"] == 10.0
