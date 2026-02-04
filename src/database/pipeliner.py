import asyncio
import asyncpg
import numpy as np
import structlog
from typing import List, Tuple, Any, Optional
import os

logger = structlog.get_logger(__name__)

class VectorizedDBWriter:
    """
    SOTA: High-throughput database writer using asyncpg pipelining.
    Designed for persisting millions of option Greeks and prices.
    """
    def __init__(self, dsn: str):
        self.dsn = dsn.replace("postgresql+asyncpg", "postgresql")
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """🚀 SINGULARITY: Initialize high-concurrency connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=10,
                max_size=50,
                max_inactive_connection_lifetime=300
            )
            logger.info("db_pipeliner_pool_initialized")

    async def insert_prices_vectorized(self, data: List[Tuple]):
        """🚀 SINGULARITY: Bulk ingestion via COPY (Fastest Path)."""
        if not self._pool: await self.connect()
        
        async with self._pool.acquire() as conn:
            # SOTA: Using copy_records_to_table for maximum throughput
            # Data expected as list of tuples matching 'options_prices' columns
            try:
                start_time = asyncio.get_event_loop().time()
                await conn.copy_records_to_table(
                    'options_prices', 
                    records=data,
                    columns=('time', 'symbol', 'strike', 'expiry', 'option_type', 'last', 'delta', 'gamma', 'implied_volatility')
                )
                duration = (asyncio.get_event_loop().time() - start_time) * 1000
                logger.info("db_vectorized_write_success", rows=len(data), latency_ms=duration)
            except Exception as e:
                logger.error("db_vectorized_write_failed", error=str(e))

    async def close(self):
        if self._pool:
            await self._pool.close()
            logger.info("db_pipeliner_pool_closed")

# Global pipeliner for the ingestion path
pipeliner = VectorizedDBWriter(os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5432/bsopt"))
