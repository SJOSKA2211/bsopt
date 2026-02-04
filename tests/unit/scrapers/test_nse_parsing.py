import pytest
from src.scrapers.engine import NSEScraper

def test_nse_cleaning_logic():
    scraper = NSEScraper()
    raw_data = {
        "price": "15.50",
        "volume": "1,200,000"
    }
    clean = scraper._clean_data(raw_data)
    assert clean["price"] == 15.5
    assert clean["volume"] == 1200000

@pytest.mark.asyncio
async def test_nse_parsing_logic():
    # Placeholder for IO test
    assert True