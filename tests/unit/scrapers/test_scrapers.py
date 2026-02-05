from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scrapers.engine import NSEScraper


@pytest.mark.asyncio
async def test_nse_scraper_http():
    # Mocking httpx responses
    mock_resp_main = MagicMock()
    # Updated to match the script regex for ajaxnonce
    mock_resp_main.text = 'var wp_ajax = {"ajaxurl":"https://www.nse.co.ke/dataservices/wp-admin/admin-ajax.php","ajaxnonce":"0c169a3698"};'
    mock_resp_main.raise_for_status = MagicMock()

    mock_resp_ajax = MagicMock()
    # Mock row matching the new site structure: Name, ISIN, Volume, Price, Change
    mock_resp_ajax.text = '<table><tr><td>Safaricom Ltd</td><td>KE1000001402</td><td>1000</td><td>29.50</td><td><span>-0.17 <i class="fa"></i></span></td></tr></table>'
    mock_resp_ajax.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.get = AsyncMock(return_value=mock_resp_main)
        mock_client.post = AsyncMock(return_value=mock_resp_ajax)
        mock_client.aclose = AsyncMock()
        
        scraper = NSEScraper()
        # Manually inject mock if constructor already ran or let patch handle it
        scraper.client = mock_client
        
        result = await scraper.get_ticker_data("SCOM")
        
        if "error" in result:
            pytest.fail(f"Scrape failed with error: {result['error']}")
            
        assert result["symbol"] == "SCOM"
        assert result["market"] == "NSE"
        assert result["price"] == 29.5
        assert result["volume"] == 1000
        assert result["change"] == -0.17