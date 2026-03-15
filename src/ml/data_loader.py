from collections.abc import Iterator
from typing import Any

import pandas as pd
import structlog

from src.ml.feature_store.store import feature_store

logger = structlog.get_logger()


class DataNormalizer:
    """
    Normalization Layer to ensure ML models receive consistent data.
    Handles synthetic bar generation via Feature Store.
    """

    @staticmethod
    def normalize_incoming(raw_data: dict[str, Any]) -> dict[str, Any]:
        """
        Converts disparate data sources into a unified OHLCV format.
        Now delegates to Feature Store logic for consistency if needed,
        but for single dicts we might keep lightweight logic or wrap in DF.
        For simplicity, we keep the lightweight logic for single-item processing here
        to avoid overhead of creating 1-row DataFrames constantly,
        but verify alignment with SyntheticOHLCFeature.
        """
        # 1. Synthetic OHLC Generation
        if "open" not in raw_data and "price" in raw_data:
            p = raw_data["price"]
            normalized = {
                "timestamp": raw_data.get("timestamp"),
                "symbol": raw_data.get("symbol"),
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "volume": raw_data.get("volume", 0),
                "source_type": "scraper_synthetic",
            }
            # Preserve other metadata
            for k, v in raw_data.items():
                if k not in normalized:
                    normalized[k] = v
            return normalized

        return raw_data

    @staticmethod
    def remove_outliers(data: dict[str, Any], prev_price: float, threshold: float = 0.1) -> bool:
        """
        Simple outlier detection logic. Returns True if data is considered an outlier.
        """
        if not prev_price:
            return False

        current_price = data.get("close") or data.get("price")
        if not current_price:
            return False

        change = abs(current_price - prev_price) / prev_price
        if change > threshold:
            logger.warning("outlier_detected", symbol=data.get("symbol"), change=change)
            return True

        return False


class StreamingDataLoader:
    """
    Optimized data loader for large datasets using chunking and generators.
    Prevents OOM errors during training by avoiding full dataset loading.
    Delegates feature engineering to the Feature Store.
    """

    def __init__(self, file_path: str, chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def stream_batches(self) -> Iterator[pd.DataFrame]:
        """
        Yields normalized dataframes in chunks with features computed.
        """
        try:
            # Use Pandas chunksize for CSVs
            if self.file_path.endswith(".csv"):
                with pd.read_csv(self.file_path, chunksize=self.chunk_size) as reader:
                    for chunk in reader:
                        yield self._process_chunk(chunk)
            elif self.file_path.endswith(".parquet"):
                df = pd.read_parquet(self.file_path)
                # Simulate chunks
                for i in range(0, len(df), self.chunk_size):
                    yield self._process_chunk(df.iloc[i : i + self.chunk_size])
        except Exception as e:
            logger.error("streaming_load_failed", error=str(e), file=self.file_path)
            raise

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize and feature engineer a chunk of data using the Feature Store.
        OPTIMIZED: Automatically uses Numba indicators if close price exists.
        """
        required_features = ["log_return"]
        if "close" in chunk.columns:
            required_features.extend(["RSI_14", "EMA_20", "MACD"])

        try:
            processed_chunk = feature_store.compute_features(chunk, required_features)
            return processed_chunk
        except Exception as e:
            logger.error("chunk_processing_error", error=str(e))
            raise e


class DatabaseDataLoader:
    """
    High-Performance ML Data Loader: Fetches training data directly from revamped hypertables.
    Uses the VectorizedDBEngine (Binary COPY/Fetch) for maximum throughput.
    """

    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = chunk_size

    async def fetch_training_set(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch a full training set for a symbol using high-speed binary retrieval.
        """
        from src.database.pipeliner import db_engine

        async with db_engine as db:
            data = await db.fetch_training_data([symbol], limit=self.chunk_size)

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            return await feature_store.compute_features(df, ["log_return"])


class AIOpsDataLoader:
    """
    Optimized loader for system metrics and audit logs.
    Fetches from revamped audit_logs and request_logs hypertables.
    """

    def __init__(self, limit: int = 10000):
        self.limit = limit

    async def fetch_system_metrics(self, hours: int = 1) -> pd.DataFrame:
        """
        Fetch latency and status code metrics for anomaly detection.
        """
        from src.database.pipeliner import db_engine

        # Optimized query using TimescaleDB hyper-functions for bucketing
        query = """
            SELECT 
                time_bucket('1 minute', created_at) AS bucket,
                AVG(duration_ms) as avg_latency,
                COUNT(*) FILTER (WHERE status_code >= 400) as error_count,
                COUNT(*) as total_requests
            FROM request_logs
            WHERE created_at > NOW() - $1::interval
            GROUP BY bucket
            ORDER BY bucket ASC
            LIMIT $2
        """

        async with db_engine as db:
            if not db._pool:
                await db.connect()

            async with db._pool.acquire() as conn:
                records = await conn.fetch(query, f"{hours} hours", self.limit)
                if not records:
                    return pd.DataFrame()
                return pd.DataFrame(records)
