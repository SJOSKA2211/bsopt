import asyncio
import time

import grpc
import structlog
import yfinance as yf
from aiolimiter import AsyncLimiter

from src.ingestion.discovery import get_sp500_symbols
from src.shared.protos import market_data_pb2, market_data_pb2_grpc

logger = structlog.get_logger(__name__)


class YFinanceScraper:
    """
    High-performance yfinance scraper for the Production universe.
    Decoupled via gRPC Ingestion Service.
    """

    def __init__(self, symbols: list[str] = None):
        self.symbols = symbols or []
        self.limiter = AsyncLimiter(max_rate=5, time_period=1.0)
        self.channel = grpc.aio.insecure_channel("localhost:50053")
        self.data_stub = market_data_pb2_grpc.DataServiceStub(self.channel)

    async def run_forever(self, interval: int = 300):
        """
        Main loop for continuous yfinance ingestion.
        """
        if not self.symbols:
            logger.info("discovering_sp500_symbols")
            self.symbols = await get_sp500_symbols()

        logger.info("yfinance_scraper_started", universe_size=len(self.symbols))

        while True:
            try:
                start_time = time.time()
                await self.scrape_universe()
                duration = time.time() - start_time
                logger.info("yfinance_universe_scrape_complete", duration=duration)

                # Robust Healthcheck Heartbeat (AIOps Compliant)
                import json

                try:
                    with open("/tmp/scraper_heartbeat", "w") as f:  # nosec B108
                        heartbeat_data = {
                            "time": time.time(),
                            "metrics": {"processed": len(self.symbols), "health": "ACTIVE"},
                        }
                        f.write(json.dumps(heartbeat_data))
                except Exception:
                    pass

                wait_time = max(0, interval - duration)
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error("yfinance_loop_error", error=str(e))
                await asyncio.sleep(60)

    async def scrape_universe(self, batch_size: int = 25):
        """
        Scrapes the entire defined universe in batches.
        """
        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i : i + batch_size]
            await self.scrape_batch(batch)
            await asyncio.sleep(0.5)  # Slight jitter

    async def scrape_batch(self, batch: list[str]):
        """
        Fetches data for a batch of symbols and pushes to gRPC.
        """
        try:
            async with self.limiter:
                # yfinance download is blocking, run in thread
                data = await asyncio.to_thread(
                    yf.download,
                    tickers=" ".join(batch),
                    period="1d",
                    interval="1m",
                    group_by="ticker",
                    threads=False,
                    progress=False,
                )

            if data.empty:
                return

            grpc_batch = []
            timestamp = int(time.time())

            if len(batch) == 1:
                sym = batch[0]
                if not data["Close"].empty:
                    grpc_batch.append(
                        market_data_pb2.Tick(
                            ticker=sym,
                            price=float(data["Close"].iloc[-1]),
                            timestamp=timestamp,
                            source="yfinance",
                        )
                    )
            else:
                for sym in batch:
                    if sym in data.columns.levels[0]:
                        sym_data = data[sym]
                        if not sym_data["Close"].empty:
                            grpc_batch.append(
                                market_data_pb2.Tick(
                                    ticker=sym,
                                    price=float(sym_data["Close"].iloc[-1]),
                                    timestamp=timestamp,
                                    source="yfinance",
                                )
                            )

            if grpc_batch:
                await self.data_stub.IngestTicks(market_data_pb2.TickBatch(ticks=grpc_batch))
                logger.debug("yfinance_batch_ingested", count=len(grpc_batch))

        except Exception as e:
            logger.error("yfinance_batch_failed", batch=batch, error=str(e))


async def main():
    # Example usage: Scrape S&P 500
    scraper = YFinanceScraper()
    await scraper.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
