import asyncio

import structlog

logger = structlog.get_logger(__name__)


class VectorizedDBEngine:
    """
    High-throughput database engine using asyncpg via SQLAlchemy's engine.
    Optimized for bulk ingestion (COPY) and fast binary retrieval.
    """

    def __init__(self):
        self._initialized = False

    async def connect(self):
        """No-op for compatibility, initialization handled by db_manager."""
        self._initialized = True

    async def _get_raw_conn(self, conn):
        """Extracts raw asyncpg connection from SQLAlchemy connection."""
        return await conn.get_raw_connection()

    async def fetch_training_data(self, symbols: list[str], limit: int = 10000) -> list[dict]:
        """
        High-speed retrieval of training data using binary format.
        """
        from src.database import db_manager
        
        # Optimized query for index-only scans
        query = """
            SELECT time, symbol, strike, expiry, option_type, last, delta, gamma, implied_volatility 
            FROM options_prices 
            WHERE symbol = ANY($1) 
            ORDER BY symbol, time DESC 
            LIMIT $2
        """
        
        async with db_manager.async_engine.connect() as conn:
            raw_conn = await self._get_raw_conn(conn)
            try:
                start_time = asyncio.get_event_loop().time()
                # Use binary format for faster transfer via raw asyncpg
                records = await raw_conn.driver_connection.fetch(query, symbols, limit)
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
        from src.database import db_manager

        async with db_manager.async_engine.connect() as conn:
            raw_conn = await self._get_raw_conn(conn)
            try:
                start_time = asyncio.get_event_loop().time()
                # asyncpg copy_records_to_table uses the binary COPY protocol by default
                await raw_conn.driver_connection.copy_records_to_table(
                    table_name, records=records, columns=columns
                )
                await conn.commit()
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


db_engine = VectorizedDBEngine()
