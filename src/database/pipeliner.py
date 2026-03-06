import asyncio

import asyncpg
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class VectorizedDBEngine:
    """
    High-throughput database engine using asyncpg.
    Optimized for bulk ingestion (COPY) and fast binary retrieval in PG16 environment.
    """

    def __init__(self, dsn: str):
        # Normalize DSN for asyncpg (remove SQLAlchemy driver prefix if present)
        self.dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Initialize the connection pool if not already active."""
        async with self._lock:
            if not self._pool:
                try:
                    # Optimized for 2GB container: restricted pool size
                    self._pool = await asyncpg.create_pool(
                        self.dsn,
                        min_size=2,
                        max_size=5,
                        command_timeout=60,
                        # Enable statement cache for repeated queries
                        max_cached_statement_lifetime=600,
                        max_cache_size=100,
                    )
                    logger.info("db_pipeliner_pool_initialized", dsn=self.dsn)
                except Exception as e:
                    logger.error("db_pipeliner_init_failed", error=str(e))
                    raise

    async def fetch_training_data(self, symbols: list[str], limit: int = 10000) -> list[dict]:
        """
        High-speed retrieval of training data using binary format.
        """
        if not self._pool:
            await self.connect()

        # Optimized query for index-only scans
        query = """
            SELECT time, symbol, strike, expiry, option_type, last, delta, gamma, implied_volatility 
            FROM options_prices 
            WHERE symbol = ANY($1) 
            ORDER BY symbol, time DESC 
            LIMIT $2
        """
        async with self._pool.acquire() as conn:
            try:
                start_time = asyncio.get_event_loop().time()
                # Use binary format for faster transfer
                records = await conn.fetch(query, symbols, limit)
                duration = (asyncio.get_event_loop().time() - start_time) * 1000
                logger.info("db_fetch_success", rows=len(records), latency_ms=round(duration, 2))
                return [dict(r) for r in records]
            except Exception as e:
                logger.error("db_fetch_failed", error=str(e))
                return []

    async def generic_bulk_copy(
        self, table_name: str, records: list[tuple], columns: tuple[str, ...]
    ):
        """
        Generic high-performance bulk insert using binary COPY.
        """
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            try:
                start_time = asyncio.get_event_loop().time()
                # asyncpg copy_records_to_table uses the binary COPY protocol by default
                await conn.copy_records_to_table(table_name, records=records, columns=columns)
                duration = (asyncio.get_event_loop().time() - start_time) * 1000
                logger.info(
                    "db_bulk_copy_success",
                    table=table_name,
                    rows=len(records),
                    latency_ms=round(duration, 2),
                )
            except Exception as e:
                logger.error(
                    "db_bulk_copy_failed",
                    table=table_name,
                    error=str(e),
                    columns=columns,
                )
                raise

    async def insert_prices_vectorized(
        self, data: list[tuple], columns: tuple[str, ...] | None = None
    ):
        """
        Specialized bulk ingestion for options prices.
        """
        target_columns = columns or (
            "time",
            "symbol",
            "strike",
            "expiry",
            "option_type",
            "last",
            "delta",
            "gamma",
            "implied_volatility",
        )
        await self.generic_bulk_copy("options_prices", data, target_columns)

    async def insert_predictions_bulk(self, predictions: list[tuple]):
        """
        Specialized bulk ingestion for ML predictions (Hypertable).
        """
        columns = ("timestamp", "symbol", "model_id", "input_features", "predicted_price")
        await self.generic_bulk_copy("model_predictions", predictions, columns)

    async def close(self):
        """Gracefully close the connection pool and release resources."""
        async with self._lock:
            if self._pool:
                await self._pool.close()
                self._pool = None
                logger.info("db_pipeliner_pool_closed")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


db_engine = VectorizedDBEngine(settings.DATABASE_URL)
