import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.scrapers.engine import NSEScraper

@pytest.fixture
def mock_playwright():
    with patch("src.scrapers.engine.async_playwright") as mock_ap:
        playwright_instance = MagicMock()
        # Mock .start() which is called in the code
        mock_ap.return_value.start = AsyncMock(return_value=playwright_instance)
        
        browser = MagicMock()
        playwright_instance.chromium.launch = AsyncMock(return_value=browser)
        
        yield playwright_instance, browser

@pytest.mark.asyncio
async def test_get_ticker_data_success(mock_playwright):
    pw, browser = mock_playwright
    scraper = NSEScraper()
    
    page = AsyncMock()
    browser.new_page = AsyncMock(return_value=page)
    
    # Mock locator: page.locator is SYNCHRONOUS
    page.locator = MagicMock()
    mock_row = MagicMock()
    page.locator.return_value = mock_row
    
    # mock_row.count is ASYNCHRONOUS
    mock_row.count = AsyncMock(return_value=1)
    
    # row.locator("...").inner_text() is ASYNCHRONOUS
    mock_col = AsyncMock()
    mock_col.inner_text.side_effect = ["150.0", "+1.5", "1000"]
    mock_row.locator.return_value = mock_col
    
    data = await scraper.get_ticker_data("SCOM")
    
    assert data["symbol"] == "SCOM"
    assert data["price"] == 150.0
    assert data["volume"] == 1000

@pytest.mark.asyncio
async def test_get_ticker_data_not_found(mock_playwright):
    pw, browser = mock_playwright
    scraper = NSEScraper()
    
    page = AsyncMock()
    browser.new_page = AsyncMock(return_value=page)
    page.locator = MagicMock()
    mock_row = MagicMock()
    page.locator.return_value = mock_row
    mock_row.count = AsyncMock(return_value=0)
    
    data = await scraper.get_ticker_data("INVALID")
    assert "error" in data
    assert data["error"] == "Ticker not found"
