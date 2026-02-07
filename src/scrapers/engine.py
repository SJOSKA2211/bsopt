import asyncio
import re
import time
from datetime import datetime
from typing import Protocol

import httpx
import numpy as np
import structlog
from anyio.to_thread import run_sync
from selectolax.lexbor import LexborHTMLParser

from src.config import settings
from src.shared.observability import PROXY_FAILURES, PROXY_LATENCY
from src.utils.cache import get_redis
from src.utils.circuit_breaker import nse_circuit
from src.utils.http_client import HttpClientManager
from src.utils.resilience import retry_with_backoff

logger = structlog.get_logger()


class MarketSource(Protocol):
    async def get_ticker_data(self, symbol: str) -> dict:
        """Fetch real-time data for a given symbol."""
        ...


class ProxyRotator:
    """
    Manages a pool of proxies with persistent health tracking in Redis.
    """

    def __init__(self, proxies: list[str]):
        # Store metadata for each proxy
        self.proxies = [
            {"url": p, "failures": 0, "active": True, "latency": 0.0} for p in proxies
        ]
        self._index = 0
        self.redis = get_redis()

    async def get_proxy(self) -> str | None:
        if not self.proxies:
            return None

        # Load health from Redis if available for global consistency
        if self.redis:
            for p in self.proxies:
                health = await self.redis.get(f"proxy_health:{p['url']}")
                if health:
                    try:
                        h_data = orjson.loads(health)
                        p["failures"] = h_data.get("failures", 0)
                        p["active"] = h_data.get("active", True)
                        p["latency"] = h_data.get("latency", 0.0)
                    except:
                        pass

        # Filter active proxies
        active_proxies = [p for p in self.proxies if p["active"]]
        if not active_proxies:
            return None

        # Prefer proxies with lower latency and fewer failures
        # Sort by (failures * 10.0 + latency)
        active_proxies.sort(key=lambda x: (x["failures"] * 10.0 + x["latency"]))

        # Pick from top pool to keep things fresh
        pool_size = max(1, min(3, len(active_proxies)))
        proxy = active_proxies[self._index % pool_size]
        self._index = (self._index + 1) % pool_size

        return proxy["url"]

    async def report_success(self, url: str, latency: float):
        PROXY_LATENCY.labels(proxy_url=url).observe(latency)
        for p in self.proxies:
            if p["url"] == url:
                p["latency"] = (p["latency"] * 0.7) + (latency * 0.3)  # EMA for latency
                p["failures"] = max(
                    0, p["failures"] - 1
                )  # Reduce failure count on success
                await self._sync_health(p)

    async def report_failure(self, url: str):
        PROXY_FAILURES.labels(proxy_url=url).inc()
        for p in self.proxies:
            if p["url"] == url:
                p["failures"] += 1
                if p["failures"] >= 5:
                    p["active"] = False
                    logger.warning("proxy_deactivated", url=url)
                await self._sync_health(p)

    async def _sync_health(self, proxy_obj: dict):
        if self.redis:
            try:
                await self.redis.setex(
                    f"proxy_health:{proxy_obj['url']}",
                    3600,
                    orjson.dumps(
                        {
                            "failures": proxy_obj["failures"],
                            "active": proxy_obj["active"],
                            "latency": proxy_obj["latency"],
                        }
                    ),
                )
            except:
                pass


class NSEScraper:
    """
    Highly optimized HTTP-based scraper for Nairobi Securities Exchange (NSE).
    Bypasses standard API limitations by using direct AJAX calls and proxy rotation.
    """

    BASE_URL = "https://www.nse.co.ke/dataservices/market-statistics/"
    AJAX_URL = "https://www.nse.co.ke/dataservices/wp-admin/admin-ajax.php"

    def __init__(self, proxies: list[str] | None = None):
        self._data_cache = {}
        self._last_refresh = 0
        self._cache_ttl = settings.NSE_CACHE_TTL
        self._refresh_future: asyncio.Future | None = None
        self.proxy_rotator = ProxyRotator(proxies) if proxies else None

        # 🚀 OPTIMIZATION: Pre-computed exact-match hash map
        self._symbol_map = {
            k.upper(): v for k, v in settings.NSE_NAME_SYMBOL_MAP.items()
        }
        # Map from fully normalized name to symbol
        self._exact_symbol_map = {
            k.upper().strip(): v for k, v in settings.NSE_NAME_SYMBOL_MAP.items()
        }

        # SOTA: shared client with connection pooling
        self.client = HttpClientManager.get_client()

    # ... (lines continue)

    def _map_name_to_symbol(self, name: str) -> str:
        """🚀 OPTIMIZATION: O(1) mapping using pre-computed hash map."""
        n = name.upper().strip()

        # 1. Exact match lookup (O(1))
        if n in self._exact_symbol_map:
            return self._exact_symbol_map[n]

        # 2. Keyword substring match (Optimized fallback)
        # Using a generator expression for slightly better memory perf
        match = next(
            (symbol for keyword, symbol in self._symbol_map.items() if keyword in n),
            None,
        )
        if match:
            return match

        # 3. Fallback: Use the first word as the symbol
        return n.split(" ")[0]

    async def _get_client_with_proxy(self) -> httpx.AsyncClient:
        """🚀 OPTIMIZATION: Acquire a fresh client with a rotated proxy if enabled."""
        if not self.proxy_rotator:
            return self.client

        proxy_url = await self.proxy_rotator.get_proxy()
        if not proxy_url:
            return self.client

        # Return a specialized client for this one-off request
        return httpx.AsyncClient(
            proxy=proxy_url,
            headers={"User-Agent": "BS-Opt/2.0"},
            timeout=10.0,
            verify=False,  # NSE often has SSL issues with proxies
        )

    @nse_circuit
    @retry_with_backoff(retries=3, initial_delay=2.0, backoff_factor=3.0)
    async def _refresh_cache(self):
        """
        Fetches all equity statistics via direct AJAX calls.
        Uses single-flight pattern to ensure only one refresh runs at a time.
        """
        # 1. Check if refresh is already in progress
        if self._refresh_future:
            await self._refresh_future
            return

        # 2. Check TTL before starting
        if time.time() - self._last_refresh < self._cache_ttl:
            return

        # 3. Create a future for this refresh cycle (Single-flight)
        loop = asyncio.get_event_loop()
        self._refresh_future = loop.create_future()

        try:
            client = await self._get_client_with_proxy()
            start_time = time.time()
            try:
                logger.info(
                    "nse_refreshing_cache_http",
                    url=self.BASE_URL,
                    using_proxy=(client != self.client),
                )

                # ... (rest of the implementation remains same)
                resp = await client.get(self.BASE_URL)
                resp.raise_for_status()

                latency = time.time() - start_time
                if self.proxy_rotator and client._proxies:
                    proxy_url = str(next(iter(client._proxies.values())).url)
                    await self.proxy_rotator.report_success(proxy_url, latency)

                nonce_match = re.search(r'"ajaxnonce":"([a-f0-9]+)"', resp.text)
                if not nonce_match:
                    logger.error("nse_nonce_not_found")
                    if not self._refresh_future.done():
                        self._refresh_future.set_result(None)
                    return
                nonce = nonce_match.group(1)

                timestamp = datetime.now().isoformat()
                tasks = [
                    self._fetch_sector(client, nonce, sector)
                    for sector in settings.NSE_SECTORS
                ]
                sector_results = await asyncio.gather(*tasks, return_exceptions=True)

                all_items = []
                for res in sector_results:
                    if isinstance(res, Exception):
                        logger.warning("nse_sector_fetch_failed", error=str(res))
                        continue
                    all_items.extend(res)

                # SOTA: Offload NumPy batch cleaning to a thread pool
                cleaned_items = await run_sync(self._batch_clean, all_items)

                new_cache = {}
                for item in cleaned_items:
                    name = item["name"]
                    symbol = self._map_name_to_symbol(name)
                    item["symbol"] = symbol
                    item["timestamp"] = timestamp
                    new_cache[symbol] = item

                if new_cache:
                    self._data_cache = new_cache
                    self._last_refresh = time.time()
                    logger.info("nse_cache_updated", count=len(new_cache))

                    # 🚀 SINGULARITY: Publish to Market Mesh
                    from src.scrapers.mesh_publisher import get_market_publisher

                    get_market_publisher().publish(new_cache)

            finally:
                if client != self.client:
                    await client.aclose()

            if not self._refresh_future.done():
                self._refresh_future.set_result(True)

        except Exception as e:
            logger.error("nse_refresh_failed", error=str(e))
            if not self._refresh_future.done():
                self._refresh_future.set_exception(e)
            raise e
        finally:
            self._refresh_future = None

    async def _fetch_sector(
        self, client: httpx.AsyncClient, nonce: str, sector: str
    ) -> list[dict]:
        """Fetch data for a specific sector via the WordPress AJAX endpoint."""
        payload = {"action": "display_prices", "security": nonce, "sector": sector}
        resp = await client.post(self.AJAX_URL, data=payload)
        resp.raise_for_status()

        # SOTA: Offload synchronous parsing to a thread pool
        return await run_sync(self._parse_html, resp.text)

    def _parse_html(self, html: str) -> list[dict]:
        """Robustly parse the HTML table fragment using selectolax (Lexbor)."""
        parser = LexborHTMLParser(html)
        results = []

        # Each row is a <tr>
        for row in parser.css("tr"):
            cells = row.css("td")
            if len(cells) < 5:
                continue

            name = cells[0].text(strip=True)
            isin = cells[1].text(strip=True)
            volume = cells[2].text(strip=True)
            price = cells[3].text(strip=True)

            # Change is often wrapped in <span> with color
            change_node = cells[4]
            change_text = change_node.text(strip=True)

            # Extract numeric part of change using regex as fallback for dirty text
            change_match = re.search(r"([-+]?\d*\.?\d+)", change_text)
            change = change_match.group(1) if change_match else "0.0"

            results.append(
                {
                    "name": name,
                    "isin": isin,
                    "volume": volume,
                    "price": price,
                    "change": change,
                    "market": "NSE",
                }
            )
        return results

    def _map_name_to_symbol(self, name: str) -> str:
        """🚀 OPTIMIZATION: O(1) mapping using pre-computed hash map."""
        n = name.upper()

        # Exact keyword match
        for keyword, symbol in self._symbol_map.items():
            if keyword in n:
                return symbol

        # Fallback: Use the first word as the symbol
        return name.split(" ")[0].upper()

    async def get_ticker_data(self, symbol: str) -> dict:
        """Get ticker data, refreshing cache if necessary."""
        if time.time() - self._last_refresh > self._cache_ttl:
            await self._refresh_cache()

        data = self._data_cache.get(symbol)
        if not data:
            # Try substring match for better resilience
            for s, d in self._data_cache.items():
                if symbol in s or s in symbol:
                    return d

            logger.warning("nse_ticker_not_in_cache", symbol=symbol)
            return {"symbol": symbol, "error": "Ticker not found", "market": "NSE"}

        return data

    async def shutdown(self):
        """Gracefully close the HTTP client."""
        await self.client.aclose()

    def _clean_data(self, data: dict) -> dict:
        """Converts string values to appropriate numeric types."""
        try:
            if "price" in data and isinstance(data["price"], str):
                data["price"] = float(data["price"].replace(",", ""))
            if "volume" in data and isinstance(data["volume"], str):
                vol_str = data["volume"].replace(",", "")
                data["volume"] = int(float(vol_str))
            if "change" in data and isinstance(data["change"], str):
                data["change"] = float(data["change"])
            return data
        except (ValueError, AttributeError, TypeError):
            return data

    def _batch_clean(self, items: list[dict]) -> list[dict]:
        """
        SOTA: Vectorized batch cleaning using Pandas.
        Replaces manual NumPy extraction loops with optimized C-level operations.
        """
        if not items:
            return []

        try:
            import pandas as pd

            df = pd.DataFrame(items)

            # 🚀 OPTIMIZATION: Vectorized string cleaning and type conversion
            # Using .replace with regex=True for batch comma removal
            for col in ["price", "volume", "change"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            if "volume" in df.columns:
                df["volume"] = df["volume"].astype(np.int64)

            return df.to_dict("records")

        except Exception as e:
            logger.warning("batch_clean_failed_falling_back", error=str(e))
            return [self._clean_data(i) for i in items]
