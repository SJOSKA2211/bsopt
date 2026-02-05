import tests.mock_all
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import httpx
from src.scrapers.engine import ProxyRotator, NSEScraper

@pytest.fixture
def mock_redis():
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.setex = AsyncMock(return_value=True)
    return mock

@pytest.mark.asyncio
async def test_proxy_rotator(mock_redis):
    proxies = ["http://p1", "http://p2"]
    with patch("src.scrapers.engine.get_redis", return_value=mock_redis):
        rotator = ProxyRotator(proxies)
        
        # Test getting proxy
        p = await rotator.get_proxy()
        assert p in proxies
        
        # Test failure reporting
        await rotator.report_failure("http://p1")
        await rotator.report_failure("http://p1")
        await rotator.report_failure("http://p1")
        await rotator.report_failure("http://p1")
        await rotator.report_failure("http://p1")
        
        # Should be deactivated
        active = [px for px in rotator.proxies if px["url"] == "http://p1"][0]
        assert active["active"] is False
        
        # Next call should not return p1
        p = await rotator.get_proxy()
        assert p == "http://p2"

def test_map_name_to_symbol():
    with patch("src.scrapers.engine.settings") as mock_settings:
        mock_settings.NSE_NAME_SYMBOL_MAP = {"Safaricom": "SCOM", "KCB": "KCB"}
        scraper = NSEScraper()
        
        assert scraper._map_name_to_symbol("Safaricom PLC") == "SCOM"
        assert scraper._map_name_to_symbol("KCB Group") == "KCB"
        assert scraper._map_name_to_symbol("UNKNOWN STOCK") == "UNKNOWN"

@pytest.mark.asyncio
async def test_refresh_cache_success():
    with patch("src.scrapers.engine.HttpClientManager.get_client") as MockClient:
        with patch("src.scrapers.engine.settings") as mock_settings:
            with patch("src.scrapers.engine.LexborHTMLParser") as MockParser:
                with patch("src.scrapers.engine.run_sync", side_effect=lambda f, *args: f(*args)):
                    
                    mock_settings.NSE_CACHE_TTL = 0
                    mock_settings.NSE_NAME_SYMBOL_MAP = {"TEST": "TST"}
                    mock_settings.NSE_SECTORS = ["Banking"]
                    
                    mock_client = MockClient.return_value
                    
                    # Mock BASE_URL response with nonce
                    mock_client.get = AsyncMock(return_value=MagicMock(
                        status_code=200, 
                        text='var nse = {"ajaxnonce":"12345"};',
                        raise_for_status=MagicMock()
                    ))
                    
                    # Mock AJAX_URL response
                    mock_client.post = AsyncMock(return_value=MagicMock(
                        status_code=200,
                        text='<table><tr><td>TEST</td><td>ISIN</td><td>100</td><td>10.0</td><td>+0.1</td></tr></table>',
                        raise_for_status=MagicMock()
                    ))
                    
                    # Mock Parser
                    mock_row = MagicMock()
                    mock_cell_name = MagicMock()
                    mock_cell_name.text.return_value = "TEST"
                    mock_cell_isin = MagicMock()
                    mock_cell_isin.text.return_value = "ISIN"
                    mock_cell_vol = MagicMock()
                    mock_cell_vol.text.return_value = "100"
                    mock_cell_price = MagicMock()
                    mock_cell_price.text.return_value = "10.0"
                    mock_cell_change = MagicMock()
                    mock_cell_change.text.return_value = "+0.1"
                    
                    mock_row.css.return_value = [mock_cell_name, mock_cell_isin, mock_cell_vol, mock_cell_price, mock_cell_change]
                    MockParser.return_value.css.return_value = [mock_row]
                    
                    scraper = NSEScraper()
                    
                    # Mock mesh publisher
                    with patch("src.scrapers.mesh_publisher.get_market_publisher"):
                        await scraper._refresh_cache()
                        
                    data = await scraper.get_ticker_data("TST")
                    assert data["price"] == 10.0
                    assert data["volume"] == 100

def test_batch_clean():
    scraper = NSEScraper()
    items = [
        {"name": "A", "price": "1,200.50", "volume": "500", "change": "+1.2"},
        {"name": "B", "price": "50", "volume": "1,000", "change": "-0.5"}
    ]
    cleaned = scraper._batch_clean(items)
    assert cleaned[0]["price"] == 1200.5
    assert cleaned[0]["volume"] == 500
    assert cleaned[1]["volume"] == 1000
