import asyncio

from src.scrapers.engine import NSEScraper


async def test():
    scraper = NSEScraper()
    print(f"Testing scraper with URL: {scraper.BASE_URL}")
    try:
        start = asyncio.get_event_loop().time()
        # Trigger cache refresh
        data = await scraper.get_ticker_data("SCOM")
        end = asyncio.get_event_loop().time()
        print(f"Refresh took {end - start:.2f}s")
        print(f"Sample data (SCOM): {data}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await scraper.shutdown()


if __name__ == "__main__":
    asyncio.run(test())
