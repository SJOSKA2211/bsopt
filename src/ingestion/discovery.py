"""
Universe Discovery Utility
==
Dynamically discovers symbols from various sources (NSE, S&P 500, etc.)
to populate the symbol universe for the ingestion pipeline.
"""

import asyncio

import httpx
import pandas as pd

from src.ingestion.engine import NSEScraper
from src.shared.observability import logger


async def get_sp500_symbols() -> list[str]:
    """Fetches S&P 500 symbols from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            df = tables[0]
            symbols = df["Symbol"].tolist()
            return [s.replace(".", "-") for s in symbols]
    except Exception as e:
        logger.error("failed_to_fetch_sp500", error=str(e))
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]  # Fallback


async def get_nse_symbols() -> list[str]:
    """Discovers all active symbols from the NSE scraper."""
    scraper = NSEScraper()
    try:
        await scraper._refresh_cache()
        return list(scraper._data_cache.keys())
    except Exception as e:
        logger.error("failed_to_discover_nse_symbols", error=str(e))
        return []
    finally:
        await scraper.shutdown()


async def discover_full_universe() -> set[str]:
    """Discovers all symbols for both markets."""
    sp500_task = asyncio.create_task(get_sp500_symbols())
    nse_task = asyncio.create_task(get_nse_symbols())

    sp500, nse = await asyncio.gather(sp500_task, nse_task)
    full_universe = set(sp500 + nse)
    logger.info("universe_discovery_complete", total_count=len(full_universe))
    return full_universe


if __name__ == "__main__":
    import json

    universe = asyncio.run(discover_full_universe())
    print(json.dumps(list(universe)))