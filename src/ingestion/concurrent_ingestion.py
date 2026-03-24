"""
Concurrent Data Ingestion Pipeline (NSE + Yahoo Finance)
========================================================
Executes high-volume asynchronous data ingestion with resilience,
rate-limiting, dynamic batching, and Pydantic normalization.
"""

import asyncio
from datetime import UTC, datetime

import structlog
import yfinance as yf
from aiolimiter import AsyncLimiter
from opentelemetry import trace
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, Field, field_validator
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from src.database import db_manager
from src.ingestion.discovery import get_sp500_symbols
from src.shared.utils.resilience import yfinance_breaker

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Prometheus Metrics
INGESTION_TICKS_TOTAL = Counter("bsopt_ingestion_ticks_total", "Total market ticks", ["market"])
INGESTION_BATCH_DURATION = Histogram("bsopt_ingestion_batch_duration_seconds", "Batch fetch time")
RATE_LIMIT_HITS = Counter("bsopt_rate_limit_hits_total", "Rate limit attempts")

yahoo_rate_limiter = AsyncLimiter(max_rate=10, time_period=1.0)

class MarketTick(BaseModel):
    symbol: str
    market: str
    price: float
    volume: int
    change: float = 0.0
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("price", "change")
    @classmethod
    def round_floats(cls, v: float) -> float:
        return round(float(v), 4)

class OptionData(BaseModel):
    symbol: str
    strike: float
    expiry: datetime
    option_type: str
    last_price: float
    bid: float
    ask: float
    implied_volatility: float
    volume: int
    open_interest: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

class SymbolMetadata(BaseModel):
    symbol: str
    name: str
    exchange: str

@yfinance_breaker
async def fetch_yfinance_batch(symbols: list[str]) -> list[MarketTick]:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    ):
        with attempt:
            RATE_LIMIT_HITS.inc()
            async with yahoo_rate_limiter:
                with (
                    INGESTION_BATCH_DURATION.time(),
                    tracer.start_as_current_span("yfinance_batch_fetch") as span,
                ):
                    span.set_attribute("symbols.count", len(symbols))

                    data = await asyncio.to_thread(
                        yf.download,
                        tickers=" ".join(symbols),
                        period="1d",
                        interval="1m",
                        group_by="ticker",
                        threads=False,
                        progress=False,
                    )

                    ticks = []
                    if data.empty:
                        return ticks

                    if len(symbols) == 1:
                        sym = symbols[0]
                        if not data["Close"].empty:
                            ticks.append(
                                MarketTick(
                                    symbol=sym,
                                    market="US",
                                    price=float(data["Close"].iloc[-1]),
                                    volume=int(data["Volume"].iloc[-1]),
                                )
                            )
                    else:
                        for sym in symbols:
                            if sym in data.columns.levels[0]:
                                sym_data = data[sym]
                                if not sym_data["Close"].empty:
                                    ticks.append(
                                        MarketTick(
                                            symbol=sym,
                                            market="US",
                                            price=float(sym_data["Close"].iloc[-1]),
                                            volume=int(sym_data["Volume"].iloc[-1]),
                                        )
                                    )
                    return ticks

async def bulk_insert_ticks(ticks: list[MarketTick]):
    if not ticks:
        return
    insert_query = """
        INSERT INTO market_ticks (time, symbol, market, price, volume, change)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (time, symbol) DO UPDATE
        SET price = EXCLUDED.price, volume = EXCLUDED.volume, change = EXCLUDED.change;
    """
    async with db_manager.async_engine.begin() as conn:
        raw_conn = await conn.get_raw_connection()
        await raw_conn.driver_connection.executemany(
            insert_query, [(t.time, t.symbol, t.market, t.price, t.volume, t.change) for t in ticks]
        )
    logger.info("bulk_insert_ticks_success", count=len(ticks))

async def bulk_insert_symbols(symbols: list[SymbolMetadata]):
    # Idempotent symbol registration
    pass

async def bulk_insert_options(options: list[OptionData]):
    # Options insertion
    pass

async def run_concurrent_ingestion(us_universe: list[str]):
    from src.streaming.rabbitmq_producer import RabbitMQMarketDataProducer

    # Discovery if needed
    if not us_universe:
        us_universe = await get_sp500_symbols()

    producer = RabbitMQMarketDataProducer()
    await producer.connect()

    try:
        # Fetch yfinance in batches to avoid OOM and hitting rate limits too hard
        batch_size = 20
        for i in range(0, len(us_universe), batch_size):
            symbols = us_universe[i : i + batch_size]
            all_ticks = await fetch_yfinance_batch(symbols)

            if all_ticks:
                # Phase 2: Decouple via RabbitMQ Topic Exchange
                # We group by symbol for the RabbitMQ payload
                tick_records = {t.symbol: t.model_dump(mode="json") for t in all_ticks}
                await producer.produce_market_data(tick_records, routing_key="us.ticks")
                logger.info("ingestion_published", count=len(all_ticks), batch=i // batch_size)

            # Rate limiting between batches
            await asyncio.sleep(1)

    finally:
        await producer.close()
        logger.info("ingestion_pipeline_complete")

if __name__ == "__main__":
    from src.shared.config import settings
    asyncio.run(run_concurrent_ingestion(settings.MARKET_TICKER_SYMBOLS))
