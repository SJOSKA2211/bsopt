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
from src.config import settings
from src.scrapers.engine import NSEScraper
from src.scrapers.discovery import get_sp500_symbols
from src.shared.observability import logger

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
            async with yahoo_rate_limiter:
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

async def yfinance_ingestion_task(universe: List[str], batch_size: int = 50) -> List[MarketTick]:
    """Manages pagination and dynamic batching."""
    all_ticks = []
    batches = [universe[i:i + batch_size] for i in range(0, len(universe), batch_size)]
    sem = asyncio.Semaphore(5)
    
    async def process_batch(batch: List[str]):
        async with sem:
            try:
                return await fetch_yfinance_batch(batch)
            except Exception as e:
                logger.error("yfinance_batch_failed", batch=batch, error=str(e))
                return []

    tasks = [process_batch(b) for b in batches]
    results = await asyncio.gather(*tasks)
    for res in results:
        all_ticks.extend(res)
    return all_ticks

async def nse_ingestion_task() -> List[MarketTick]:
    """NSE scraper integration."""
    scraper = NSEScraper()
    ticks = []
    try:
        await scraper._refresh_cache()
        for symbol, data in scraper._data_cache.items():
            try:
                ticks.append(MarketTick(
                    symbol=symbol,
                    market="NSE",
                    price=float(data.get("price", 0.0)),
                    volume=int(data.get("volume", 0)),
                    change=float(data.get("change", 0.0))
                ))
            except Exception:
                pass
    except Exception as e:
        logger.error("nse_ingestion_failed", error=str(e))
    finally:
        await scraper.shutdown()
    return ticks

# ─── Bulk Insertion ──────────────────────────────────────────────────────────

def get_db_engine():
    """Helper to get async engine with pgbouncer awareness."""
    db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    pool_kwargs = {}
    if settings.PGBOUNCER_ENABLED:
        pool_kwargs["poolclass"] = NullPool
    else:
        pool_kwargs["pool_size"] = 10
        pool_kwargs["max_overflow"] = 20
        
    return create_async_engine(db_url, **pool_kwargs)

async def bulk_insert_ticks(ticks: List[MarketTick]):
    """Bulk inserts market ticks into TimescaleDB."""
    if not ticks: return
    engine = get_db_engine()
    
    records = [(t.time, t.symbol, t.market, t.price, t.volume, t.change) for t in ticks]
    insert_query = """
        INSERT INTO market_ticks (time, symbol, market, price, volume, change)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (time, symbol) DO UPDATE
        SET price = EXCLUDED.price, volume = EXCLUDED.volume, change = EXCLUDED.change;
    """
    try:
        async with engine.begin() as conn:
            raw_conn = await conn.get_raw_connection()
            await raw_conn.driver_connection.executemany(insert_query, records)
        logger.info("bulk_insert_ticks_success", count=len(records))
    finally:
        await engine.dispose()

async def bulk_insert_symbols(symbols: List[SymbolMetadata]):
    """Bulk inserts symbol metadata."""
    if not symbols: return
    engine = get_db_engine()
    
    records = [(s.symbol, s.name, s.exchange, s.sector, s.industry, s.market_cap, s.is_active, datetime.utcnow()) for s in symbols]
    insert_query = """
        INSERT INTO symbols (symbol, name, exchange, sector, industry, market_cap, is_active, last_updated)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (symbol) DO UPDATE
        SET name = EXCLUDED.name, sector = COALESCE(EXCLUDED.sector, symbols.sector), industry = COALESCE(EXCLUDED.industry, symbols.industry),
            market_cap = COALESCE(EXCLUDED.market_cap, symbols.market_cap), last_updated = EXCLUDED.last_updated;
    """
    try:
        async with engine.begin() as conn:
            raw_conn = await conn.get_raw_connection()
            await raw_conn.driver_connection.executemany(insert_query, records)
        logger.info("bulk_insert_symbols_success", count=len(records))
    finally:
        await engine.dispose()

# ─── Orchestrator ────────────────────────────────────────────────────────────

async def run_concurrent_ingestion(us_universe: Optional[List[str]] = None):
    """Main orchestrator."""
    logger.info("concurrent_ingestion_started")
    start_time = time.time()
    
    if us_universe is None:
        logger.info("discovering_us_universe")
        us_universe = await get_sp500_symbols()
    
    nse_task = asyncio.create_task(nse_ingestion_task())
    us_ticks = []
    if YFINANCE_AVAILABLE:
        us_task = asyncio.create_task(yfinance_ingestion_task(us_universe))
        nse_ticks, us_ticks = await asyncio.gather(nse_task, us_task)
    else:
        nse_ticks = await nse_task
    
    all_ticks = nse_ticks + us_ticks
    logger.info("ingestion_completed", total_ticks=len(all_ticks))
    
    # Symbols Metadata
    symbols_meta = []
    seen = set()
    for t in all_ticks:
        if t.symbol not in seen:
            symbols_meta.append(SymbolMetadata(symbol=t.symbol, name=t.symbol, exchange=t.market))
            seen.add(t.symbol)

    await asyncio.gather(bulk_insert_ticks(all_ticks), bulk_insert_symbols(symbols_meta))
    logger.info("ingestion_pipeline_complete", duration=round(time.time()-start_time, 2))

if __name__ == "__main__":
    asyncio.run(run_concurrent_ingestion())
