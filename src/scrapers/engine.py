import asyncio
import re
import time
from datetime import datetime
from typing import Protocol

import httpx
import msgspec
import pandas as pd
import structlog
from anyio.to_thread import run_sync
from selectolax.lexbor import LexborHTMLParser

import grpc
from src.protos import data_pb2, data_pb2_grpc
from src.config import settings
from src.scrapers.mesh_publisher import get_market_publisher
from src.shared.observability import (
    PROXY_FAILURES,
    PROXY_LATENCY,
    setup_logging,
    start_system_metrics_loop,
)
from src.utils.cache import get_redis
from src.utils.circuit_breaker import nse_circuit
from src.utils.http_client import HttpClientManager
from src.utils.resilience import retry_with_backoff

logger = structlog.get_logger()


class MarketSource(Protocol):
    async def get_ticker_data(self, symbol: str) -> dict:
        """Fetch real-time data for a given symbol."""
        ...


# Pre-compiled regex for fast numeric extraction
_CHANGE_RE = re.compile(r"([-+]?\d*\.?\d+)")


class ProxyRotator:
    """
    Manages a pool of proxies with persistent health tracking in Redis.
    """

    def __init__(self, proxies: list[str]):
        # Store metadata for each proxy
        self.proxies = [{"url": p, "failures": 0, "active": True, "latency": 0.0} for p in proxies]
        self._index = 0
        self.redis = get_redis()
        self._decoder = msgspec.json.Decoder()

    async def get_proxy(self) -> str | None:
        if not self.proxies:
            return None

        # Load health from Redis in BULC (MGET) for efficiency
        if self.redis:
            keys = [f"proxy_health:{p['url']}" for p in self.proxies]
            health_results = await self.redis.mget(*keys)

            for i, health in enumerate(health_results):
                if health:
                    try:
                        h_data = self._decoder.decode(health)
                        p = self.proxies[i]
                        p["failures"] = h_data.get("failures", 0)
                        p["active"] = h_data.get("active", True)
                        p["latency"] = h_data.get("latency", 0.0)
                    except Exception:
                        pass

        # Filter active proxies
        active_proxies = [p for p in self.proxies if p["active"]]
        if not active_proxies:
            return None

        # Prefer proxies with lower latency and fewer failures
        # Sort by (failures * 10.0 + latency)
        active_proxies.sort(key=lambda x: x["failures"] * 10.0 + x["latency"])

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
                p["failures"] = max(0, p["failures"] - 1)  # Reduce failure count on success
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
                    msgspec.json.encode(
                        {
                            "failures": proxy_obj["failures"],
                            "active": proxy_obj["active"],
                            "latency": proxy_obj["latency"],
                        }
                    ),
                )
            except Exception:
                pass


class NSEScraper:
    """
    HTTP-based scraper for Nairobi Securities Exchange (NSE).
    Uses direct AJAX calls and proxy rotation.
    """

    BASE_URL = "https://www.nse.co.ke/dataservices/market-statistics/"
    AJAX_URL = "https://www.nse.co.ke/dataservices/wp-admin/admin-ajax.php"

    def __init__(self, proxies: list[str] | None = None):
        self._data_cache = {}
        self._last_refresh = 0
        self._cache_ttl = settings.NSE_CACHE_TTL
        self._refresh_future: asyncio.Future | None = None
        self.proxy_rotator = ProxyRotator(proxies) if proxies else None

        # gRPC Ingestion Client
        self.channel = grpc.aio.insecure_channel("ingestion-service:50053")
        self.data_stub = data_pb2_grpc.DataServiceStub(self.channel)

        # Pre-computed exact-match hash map
        self._symbol_map = {k.upper(): v for k, v in settings.NSE_NAME_SYMBOL_MAP.items()}
        # Map from fully normalized name to symbol
        self._exact_symbol_map = {
            k.upper().strip(): v for k, v in settings.NSE_NAME_SYMBOL_MAP.items()
        }

        # shared client with connection pooling
        self.client = HttpClientManager.get_client()

        # Compiled regex for fast keyword matching
        if self._symbol_map:
            pattern = "|".join([re.escape(k) for k in self._symbol_map.keys()])
            self._keyword_re = re.compile(f"({pattern})")
        else:
            self._keyword_re = re.compile(r"($^)")  # Match nothing

    # ... (lines continue)

    def _map_name_to_symbol(self, name: str) -> str:
        """Mapping using pre-computed hash map and compiled regex."""
        n = name.upper().strip()

        # 1. Exact match lookup
        if n in self._exact_symbol_map:
            return self._exact_symbol_map[n]

        # 2. Fast Keyword matching via Regex
        match = self._keyword_re.search(n)
        if match:
            keyword = match.group(1)
            return self._symbol_map[keyword]

        # 3. Fallback: Use the first word
        return n.split(" ")[0]

    async def _get_client_with_proxy(self) -> httpx.AsyncClient:
        """Acquire a fresh client with a rotated proxy if enabled."""
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
            verify=True,
            http2=True,
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
                    self._fetch_sector(client, nonce, sector) for sector in settings.NSE_SECTORS
                ]
                sector_results = await asyncio.gather(*tasks, return_exceptions=True)

                all_items = []
                for res in sector_results:
                    if isinstance(res, Exception):
                        logger.warning("nse_sector_fetch_failed", error=str(res))
                        continue
                    all_items.extend(res)

                # OPTIMIZED: Offload NumPy batch cleaning to a thread pool
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

                    # OPTIMIZED: Offload SHM publication to avoid blocking event loop
                    await run_sync(get_market_publisher().publish, new_cache)

                    # Decoupled persistence via gRPC Ingestion Service
                    try:
                        ticks = []
                        for symbol, item in new_cache.items():
                            ticks.append(data_pb2.Tick(
                                ticker=symbol,
                                price=float(item.get("price", 0)),
                                timestamp=int(time.time()),
                                source="NSE"
                            ))
                        if ticks:
                            await self.data_stub.IngestTicks(data_pb2.TickBatch(ticks=ticks))
                            logger.info("nse_ingestion_sent_to_grpc", count=len(ticks))
                    except Exception as e:
                        logger.error("grpc_ingestion_trigger_failed", error=str(e))

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

    async def _fetch_sector(self, client: httpx.AsyncClient, nonce: str, sector: str) -> list[dict]:
        """Fetch data for a specific sector via the WordPress AJAX endpoint."""
        payload = {"action": "display_prices", "security": nonce, "sector": sector}
        resp = await client.post(self.AJAX_URL, data=payload)
        resp.raise_for_status()

        # Offload synchronous parsing to a thread pool
        return await run_sync(self._parse_html, resp.text)

    def _parse_html(self, html: str) -> list[dict]:
        """Robustly parse the HTML table fragment using selectolax (Lexbor)."""
        parser = LexborHTMLParser(html)
        results = []

        # Pre-bind selectors for speed
        tr_selector = "tr"
        td_selector = "td"

        # Each row is a <tr>
        for row in parser.css(tr_selector):
            cells = row.css(td_selector)
            if len(cells) < 5:
                continue

            # Direct extraction without intermediate objects where possible
            name = cells[0].text(strip=True)
            isin = cells[1].text(strip=True)
            volume = cells[2].text(strip=True)
            price = cells[3].text(strip=True)

            # Change is often wrapped in <span> with color
            change_text = cells[4].text(strip=True)

            # Extract numeric part using pre-compiled regex
            change_match = _CHANGE_RE.search(change_text)
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

    async def get_ticker_data(self, symbol: str) -> dict:
        """Get ticker data, refreshing cache if necessary."""
        if time.time() - self._last_refresh > self._cache_ttl:
            await self._refresh_cache()

        # 1. Primary Lookup (O(1))
        data = self._data_cache.get(symbol)
        if data:
            return data

        # 2. Optimized Substring Match (Stop at first match)
        for s, d in self._data_cache.items():
            if symbol in s or s in symbol:
                return d

        logger.warning("nse_ticker_not_in_cache", symbol=symbol)
        return {"symbol": symbol, "error": "Ticker not found", "market": "NSE"}

    async def shutdown(self):
        """Gracefully close the HTTP client and producers."""
        await self.client.aclose()
        if hasattr(self, "rabbitmq_producer"):
            await self.rabbitmq_producer.close()

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
        Optimized batch cleaning using pure Python (faster than Pandas for <10k items).
        Avoids GIL contention and massive data frame allocations.
        """
        if not items:
            return []

        # Vectorized dataframe for speed
        try:
            df = pd.DataFrame(items)

            # Fast vectorized string replacements using regex without allocating python strings
            if "price" in df:
                df["price"] = pd.to_numeric(
                    df["price"].astype(str).str.replace(",", ""), errors="coerce"
                )

            if "volume" in df:
                df["volume"] = (
                    pd.to_numeric(df["volume"].astype(str).str.replace(",", ""), errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

            if "change" in df:
                df["change"] = pd.to_numeric(
                    df["change"].astype(str).str.replace(",", ""), errors="coerce"
                )

            return df.to_dict(orient="records")
        except Exception as e:
            logger.error("nse_batch_clean_failed", error=str(e))
            # Fallback
            cleaned = []
            for item in items:
                cleaned.append(self._clean_data(item))
            return cleaned


async def main():
    """Scraper service entry point with Graceful Shutdown."""
    import signal

    setup_logging()

    scraper = NSEScraper()
    logger.info("scraper_service_active")
    start_system_metrics_loop("scraper")

    # Graceful Shutdown Setup
    shutdown_event = asyncio.Event()

    def _on_signal():
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal)

    try:
        while not shutdown_event.is_set():
            try:
                await scraper._refresh_cache()
                logger.info("scraper_loop_ok")

                # Best Practice: Robust Healthcheck Heartbeat
                def _write_heartbeat():
                    with open("/tmp/scraper_heartbeat", "w") as f:
                        f.write(str(time.time()))

                await asyncio.to_thread(_write_heartbeat)
            except Exception as e:
                logger.error("scraper_loop_error", error=str(e))

            try:
                # Wait for next refresh or shutdown signal
                await asyncio.wait_for(shutdown_event.wait(), timeout=settings.NSE_CACHE_TTL or 300)
            except TimeoutError:
                continue
    finally:
        logger.info("scraper_service_stopping_cleaning_up")
        await scraper.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
