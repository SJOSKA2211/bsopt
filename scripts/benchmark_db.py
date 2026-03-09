"""
Database Benchmarking Suite (The Optimizer - God Mode)
Measures latency and throughput of core database operations in BS-OPT.
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import db_manager

logger = structlog.get_logger(__name__)

# --- CONFIGURATION ---
BATCH_SIZE = 10000
SYMBOLS = ["BTC/USD", "ETH/USD", "AAPL", "TSLA", "MSFT"]


async def benchmark_ingestion(session: AsyncSession):
    """Measures throughput of large-scale ingestion."""
    logger.info("benchmarking_ingestion_start", batch_size=BATCH_SIZE)

    # We'll use a raw asyncpg connection for maximum COPY performance
    conn = await session.connection()
    raw_conn = await conn.get_raw_connection()

    # Prepare data
    now = datetime.now(UTC)
    data = []
    for i in range(BATCH_SIZE):
        data.append(
            (
                now - timedelta(seconds=i),
                SYMBOLS[i % len(SYMBOLS)],
                float(np.random.uniform(100, 50000)),
                int(np.random.randint(1, 100)),
                "buy" if i % 2 == 0 else "sell",
            )
        )

    start_time = time.time()
    # TimescaleDB optimized ingestion
    await raw_conn.driver_connection.copy_records_to_table(
        "market_ticks", records=data, columns=["time", "symbol", "price", "volume", "side"]
    )
    duration = time.time() - start_time
    throughput = BATCH_SIZE / duration

    logger.info(
        "benchmarking_ingestion_complete",
        duration_s=round(duration, 4),
        throughput_rps=round(throughput, 2),
    )
    return throughput


async def benchmark_complex_query(session: AsyncSession):
    """Measures latency of complex Greeks-based queries with joins."""
    logger.info("benchmarking_complex_query_start")

    query = text("""
        SELECT 
            symbol,
            time_bucket('1 minute', time) as bucket,
            avg(implied_volatility) as avg_iv,
            avg(delta) as avg_delta,
            max(last) as max_price
        FROM options_prices
        WHERE time > NOW() - INTERVAL '1 hour'
        GROUP BY symbol, bucket
        ORDER BY bucket DESC
        LIMIT 100;
    """)

    start_time = time.time()
    # We run it multiple times to get a stable average
    iterations = 10
    for _ in range(iterations):
        await session.execute(query)

    duration = (time.time() - start_time) / iterations
    logger.info("benchmarking_complex_query_complete", avg_latency_ms=round(duration * 1000, 2))
    return duration


async def benchmark_cagg_refresh(session: AsyncSession):
    """Measures performance of TimescaleDB continuous aggregate refreshes."""
    logger.info("benchmarking_cagg_refresh_start")

    start_time = time.time()
    # Note: Refreshes only work if there's data in the refresh window
    try:
        await session.execute(
            text("CALL refresh_continuous_aggregate('minute_stats_cagg', NULL, NULL);")
        )
    except Exception as e:
        logger.warning("cagg_refresh_failed_or_no_cagg", error=str(e))
        return 0

    duration = time.time() - start_time
    logger.info("benchmarking_cagg_refresh_complete", duration_s=round(duration, 4))
    return duration


async def run_suite():
    """Executes the full benchmark suite."""
    logger.info("starting_full_benchmark_suite", environment="test")

    db_manager.initialize()

    async with db_manager.async_session_factory() as session:
        # 0. Setup: Ensure tables exist
        await session.execute(text("SELECT 1"))

        # 1. Ingestion
        ingest_tput = await benchmark_ingestion(session)

        # 2. Querying (Insert some dummy data first to avoid empty results)
        now = datetime.now(UTC)
        await session.execute(
            text("""
            INSERT INTO options_prices (time, symbol, strike, expiry, option_type, last, implied_volatility, delta)
            VALUES (:t, 'BTC/USD', 50000, '2026-12-31', 'call', 2500, 0.5, 0.6)
            ON CONFLICT DO NOTHING
        """),
            {"t": now},
        )
        await session.commit()

        query_latency = await benchmark_complex_query(session)

        # 3. Aggregates
        cagg_duration = await benchmark_cagg_refresh(session)

        logger.info(
            "benchmark_suite_summary",
            ingestion_throughput_rps=round(ingest_tput, 2),
            query_latency_ms=round(query_latency * 1000, 2),
            cagg_refresh_s=round(cagg_duration, 4),
        )


if __name__ == "__main__":
    asyncio.run(run_suite())
