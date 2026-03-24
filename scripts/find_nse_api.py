import asyncio
import os
from urllib.parse import parse_qs

from playwright.async_api import async_playwright

TEMP_DIR = (
    "/home/kamau/.gemini/tmp/2b8aa2f42273da2920d6a0846a0beee0179039373f45f52c93793c0248598408"
)

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        async def handle_response(response):
            if "admin-ajax.php" in response.url:
                request = response.request
                post_data = request.post_data
                action = "unknown"
                if post_data:
                    params = parse_qs(post_data)
                    action = params.get("action", ["unknown"])[0]
                    sector = params.get("sector", ["none"])[0]
                    action = f"{action}_{sector}"

                try:
                    import anyio

                    body = await response.text()
                    fname = f"nse_ajax_{action}.html"
                    fpath = os.path.join(TEMP_DIR, fname)

                    def write_file():
                        with open(fpath, "w") as f:
                            f.write(body)

                    await anyio.to_thread.run_sync(write_file)
                    print(f"Saved {fname}")
                except Exception:
                    pass

        page.on("response", handle_response)
        print("Navigating to NSE Data Services...")
        await page.goto(
            "https://www.nse.co.ke/dataservices/market-statistics/",
            wait_until="networkidle",
        )
        await asyncio.sleep(20)  # Long wait for all sectors
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
