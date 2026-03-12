"""
Concurrent Data Ingestion Pipeline (NSE + Yahoo Finance)
========================================================
Executes high-volume asynchronous data ingestion with resilience,
rate-limiting, dynamic batching, and Pydantic normalization.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from prometheus_client import Counter, Histogram, Gauge

from src.config import settings
from src.scrapers.engine import NSEScraper
from src.scrapers.discovery import get_sp500_symbols
from src.shared.observability import logger

# Prometheus Metrics
INGESTION_TICKS_TOTAL = Counter("bsopt_ingestion_ticks_total", "Total number of market ticks ingested", ["market"])
INGESTION_OPTIONS_TOTAL = Counter("bsopt_ingestion_options_total", "Total number of option ticks ingested")
INGESTION_BATCH_DURATION = Histogram("bsopt_ingestion_batch_duration_seconds", "Time spent fetching a batch of data")
DB_INSERT_DURATION = Histogram("bsopt_db_insert_duration_seconds", "Time spent bulk inserting to DB", ["table"])
INGESTION_ERRORS = Counter("bsopt_ingestion_errors_total", "Total ingestion errors", ["type"])
RATE_LIMIT_HITS = Counter("bsopt_rate_limit_hits_total", "Total rate limit / backoff attempts")
ACTIVE_INGESTION_TASKS = Gauge("bsopt_active_ingestion_tasks", "Number of concurrent ingestion tasks")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from aiolimiter import AsyncLimiter
except ImportError:
    class AsyncLimiter:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass

try:
    from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
except ImportError:
    class AsyncRetrying:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def __aiter__(self):
            async def gen(): yield self
            return gen()

# ─── Pydantic Data Normalization Layer ──────────────────────────────────────

class SymbolMetadata(BaseModel):
    """Normalized Symbol Metadata for categorization."""
    symbol: str
    name: str
    exchange: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    is_active: bool = True

class MarketTick(BaseModel):
    """Normalized Market Tick for bulk insertion into PostgreSQL."""
    symbol: str
    market: str
    price: float
    volume: int
    change: float = 0.0
    time: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("price", "change")
    @classmethod
    def round_floats(cls, v: float) -> float:
        return round(float(v), 4)

    @field_validator("volume")
    @classmethod
    def ensure_positive_volume(cls, v: int) -> int:
        return max(0, v)


class OptionData(BaseModel):
    """Normalized Option Data."""
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    last_price: float
    bid: float
    ask: float
    implied_volatility: float
    volume: int
    open_interest: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ─── Resilience & Rate Limiting ──────────────────────────────────────────────

yahoo_rate_limiter = AsyncLimiter(max_rate=10, time_period=1.0)

async def fetch_yfinance_batch(symbols: List[str]) -> List[MarketTick]:
    """
    Fetches a batch of symbols from yfinance using async thread pool.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    ):
        with attempt:
            RATE_LIMIT_HITS.inc()
            async with yahoo_rate_limiter:
                with INGESTION_BATCH_DURATION.time():
                    logger.info("yfinance_batch_fetch_start", symbols=symbols)
                data = await asyncio.to_thread(
                    yf.download, 
                    tickers=" ".join(symbols),
                    period="1d",
                    interval="1m",
                    group_by="ticker",
                    threads=False,
                    progress=False
                )
                
                ticks = []
                if data.empty:
                    return ticks

                if len(symbols) == 1:
                    sym = symbols[0]
                    if not data["Close"].empty:
                        close = float(data["Close"].iloc[-1])
                        vol = int(data["Volume"].iloc[-1])
                        ticks.append(MarketTick(symbol=sym, market="US", price=close, volume=vol))
                else:
                    for sym in symbols:
                        if sym in data.columns.levels[0]:
                            sym_data = data[sym]
                            if not sym_data["Close"].empty:
                                close = float(sym_data["Close"].iloc[-1])
                                vol = int(sym_data["Volume"].iloc[-1])
                                ticks.append(MarketTick(symbol=sym, market="US", price=close, volume=vol))

                return ticks

async def fetch_options_chain(symbol: str) -> List[OptionData]:
    """Fetches option chains for a symbol from yfinance."""
    async with yahoo_rate_limiter:
        try:
            ticker = yf.Ticker(symbol)
            # Fetch available expiries
            expiries = await asyncio.to_thread(lambda: ticker.options)
            if not expiries:
                return []
            
            # For "comprehensive" data, we take the first 3 expiries to avoid massive latency
            all_options = []
            for expiry in expiries[:3]:
                chain = await asyncio.to_thread(ticker.option_chain, expiry)
                
                for _, row in chain.calls.iterrows():
                    all_options.append(OptionData(
                        symbol=symbol,
                        strike=float(row['strike']),
                        expiry=datetime.strptime(expiry, '%Y-%m-%d'),
                        option_type='call',
                        last_price=float(row['lastPrice']),
                        bid=float(row['bid']),
                        ask=float(row['ask']),
                        implied_volatility=float(row['impliedVolatility']),
                        volume=int(row.get('volume', 0) or 0),
                        open_interest=int(row.get('openInterest', 0) or 0)
                    ))
                
                for _, row in chain.puts.iterrows():
                    all_options.append(OptionData(
                        symbol=symbol,
                        strike=float(row['strike']),
                        expiry=datetime.strptime(expiry, '%Y-%m-%d'),
                        option_type='put',
                        last_price=float(row['lastPrice']),
                        bid=float(row['bid']),
                        ask=float(row['ask']),
                        implied_volatility=float(row['impliedVolatility']),
                        volume=int(row.get('volume', 0) or 0),
                        open_interest=int(row.get('openInterest', 0) or 0)
                    ))
            return all_options
        except Exception as e:
            logger.warning("options_fetch_failed", symbol=symbol, error=str(e))
            return []

async def yfinance_ingestion_task(universe: List[str], batch_size: int = 50) -> tuple[List[MarketTick], List[OptionData]]:
    """Manages pagination and dynamic batching."""
    all_ticks = []
    all_options = []
    batches = [universe[i:i + batch_size] for i in range(0, len(universe), batch_size)]
    sem = asyncio.Semaphore(5)
    
    async def process_batch(batch: List[str]):
        async with sem:
            try:
                ticks = await fetch_yfinance_batch(batch)
                # For each symbol in batch, concurrently fetch options if it's one of the first few
                # In a real "aggressive" scenario, we'd do all, but here we limit for stability
                opt_tasks = [fetch_options_chain(s) for s in batch[:10]] 
                options_results = await asyncio.gather(*opt_tasks)
                opts = [o for sublist in options_results for o in sublist]
                return ticks, opts
            except Exception as e:
                logger.error("yfinance_batch_failed", batch=batch, error=str(e))
                return [], []

    tasks = [process_batch(b) for b in batches]
    results = await asyncio.gather(*tasks)
    for ticks, opts in results:
        all_ticks.extend(ticks)
        all_options.extend(opts)
    return all_ticks, all_options

async def nse_ingestion_task() -> List[MarketTick]:
    """NSE scraper integration (Optimized)."""
    scraper = NSEScraper()
    ticks = []
    try:
        await scraper._refresh_cache()
        # Vectorized processing of NSE cache
        for symbol, data in scraper._data_cache.items():
            try:
                # Capture NSE specific fields
                ticks.append(MarketTick(
                    symbol=symbol,
                    market="NSE",
                    price=float(data.get("price", 0.0)),
                    volume=int(data.get("volume", 0)),
                    change=float(data.get("change", 0.0))
                ))
            except (ValueError, TypeError):
                logger.debug("invalid_nse_data_skipping", symbol=symbol)
                continue
    except Exception as e:
        logger.error("nse_ingestion_failed", error=str(e))
    finally:
        await scraper.shutdown()
    return ticks

# ─── Bulk Insertion ──────────────────────────────────────────────────────────

def get_db_engine():
    """Helper to get async engine with pgbouncer awareness."""
    db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    pool_kwargs: dict[str, Any] = {}
    if settings.PGBOUNCER_ENABLED:
        pool_kwargs["poolclass"] = NullPool
    else:
        pool_kwargs["pool_size"] = 20
        pool_kwargs["max_overflow"] = 40
        
    return create_async_engine(db_url, **pool_kwargs)

async def bulk_insert_ticks(ticks: List[MarketTick]):
    """Bulk inserts market ticks into PostgreSQL (Optimized for asyncpg)."""
    if not ticks: return
    engine = get_db_engine()
    
    # Precise ordering for bulk COPY or executemany
    records = [(t.time, t.symbol, t.market, float(t.price), int(t.volume), float(t.change)) for t in ticks]
    
    insert_query = """
        INSERT INTO market_ticks (time, symbol, market, price, volume, change)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (time, symbol) DO UPDATE
        SET price = EXCLUDED.price, volume = EXCLUDED.volume, change = EXCLUDED.change;
    """
    
    try:
        with DB_INSERT_DURATION.labels(table="market_ticks").time():
            async with engine.begin() as conn:
                # Drop to the raw driver connection for maximum throughput
                raw_conn = await conn.get_raw_connection()
                # asyncpg's executemany is significantly faster than SQLAlchemy's for large batches
                await raw_conn.driver_connection.executemany(insert_query, records)
        INGESTION_TICKS_TOTAL.labels(market="US").inc(len(records))
        logger.info("bulk_insert_ticks_success", count=len(records))
    except Exception as e:
        INGESTION_ERRORS.labels(type="db_insert_ticks").inc()
        logger.error("bulk_insert_ticks_failed", error=str(e))
        raise
    finally:
        await engine.dispose()

async def bulk_insert_symbols(symbols: List[SymbolMetadata]):
    """Bulk inserts symbol metadata (idempotent)."""
    if not symbols: return
    engine = get_db_engine()
    
    records = [
        (s.symbol, s.name, s.exchange, s.sector, s.industry, s.market_cap, s.is_active, datetime.utcnow()) 
        for s in symbols
    ]
    
    insert_query = """
        INSERT INTO symbols (symbol, name, exchange, sector, industry, market_cap, is_active, last_updated)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (symbol) DO UPDATE
        SET name = EXCLUDED.name, 
            sector = COALESCE(EXCLUDED.sector, symbols.sector), 
            industry = COALESCE(EXCLUDED.industry, symbols.industry),
            market_cap = COALESCE(EXCLUDED.market_cap, symbols.market_cap), 
            last_updated = EXCLUDED.last_updated;
    """
    
    try:
        async with engine.begin() as conn:
            raw_conn = await conn.get_raw_connection()
            await raw_conn.driver_connection.executemany(insert_query, records)
        logger.info("bulk_insert_symbols_success", count=len(records))
    except Exception as e:
        logger.error("bulk_insert_symbols_failed", error=str(e))
    finally:
        await engine.dispose()

# ─── Orchestrator ────────────────────────────────────────────────────────────

async def bulk_insert_options(options: List[OptionData]):
    """Bulk inserts option chains into PostgreSQL."""
    if not options: return
    engine = get_db_engine()
    
    records = [
        (o.symbol, o.strike, o.expiry, o.option_type, float(o.last_price), 
         float(o.bid), float(o.ask), float(o.implied_volatility), int(o.volume), 
         int(o.open_interest), o.timestamp) 
        for o in options
    ]
    
    insert_query = """
        INSERT INTO option_ticks (
            symbol, strike, expiry, option_type, last_price, 
            bid, ask, implied_volatility, volume, open_interest, time
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (symbol, strike, expiry, option_type, time) DO NOTHING;
    """
    
    try:
        async with engine.begin() as conn:
            raw_conn = await conn.get_raw_connection()
            await raw_conn.driver_connection.executemany(insert_query, records)
        logger.info("bulk_insert_options_success", count=len(records))
    except Exception as e:
        logger.error("bulk_insert_options_failed", error=str(e))
    finally:
        await engine.dispose()

# ─── Orchestrator ────────────────────────────────────────────────────────────

async def run_concurrent_ingestion(us_universe: Optional[List[str]] = None):
    """Main orchestrator for high-volume data ingestion."""
    logger.info("ingestion_pipeline_start")
    start_time = time.time()
    
    if us_universe is None:
        logger.info("starting_discovery")
        us_universe = await get_sp500_symbols()
    
    # Concurrently execute NSE and yfinance tasks
    nse_task = asyncio.create_task(nse_ingestion_task())
    us_ticks_result = []
    us_options_result = []
    
    if YFINANCE_AVAILABLE:
        us_task = asyncio.create_task(yfinance_ingestion_task(us_universe))
        (nse_ticks), (us_ticks_result, us_options_result) = await asyncio.gather(nse_task, us_task)
    else:
        nse_ticks = await nse_task
    
    all_ticks = nse_ticks + us_ticks_result
    logger.info("scrapers_finished", nse_count=len(nse_ticks), us_count=len(us_ticks_result), options_count=len(us_options_result))
    
    # Extract unique symbols for metadata registration
    symbols_meta = []
    seen_symbols = set()
    for t in all_ticks:
        if t.symbol not in seen_symbols:
            symbols_meta.append(SymbolMetadata(
                symbol=t.symbol, 
                name=t.symbol, 
                exchange=t.market
            ))
            seen_symbols.add(t.symbol)

    # Concurrently register symbols and insert time-series ticks
    await asyncio.gather(
        bulk_insert_ticks(all_ticks), 
        bulk_insert_symbols(symbols_meta),
        bulk_insert_options(us_options_result)
    )
    
    duration = round(time.time() - start_time, 2)
    logger.info("ingestion_pipeline_complete", duration_seconds=duration, total_ticks=len(all_ticks), total_options=len(us_options_result))

if __name__ == "__main__":
    import structlog
    import os
    from prometheus_client import start_http_server
    
    structlog.configure()
    
    # Start Prometheus metrics server on a configurable port
    metrics_port = int(os.getenv("METRICS_PORT", "8001"))
    start_http_server(metrics_port)
    logger.info("metrics_server_started", port=metrics_port)
    
    asyncio.run(run_concurrent_ingestion())
