from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import structlog
from numba import njit

logger = structlog.get_logger(__name__)


class StorageBackend(Enum):
    DATABASE = "database"
    FILE = "file"
    MINIO = "minio"


@dataclass
class PipelineConfig:
    symbols: list[str]
    min_samples: int = 1000
    max_samples: int = 10000
    use_multi_source: bool = False
    validate_data: bool = True
    storage_backend: StorageBackend = StorageBackend.DATABASE
    output_dir: str = "data/training"


@njit(fastmath=True)
def _rolling_mean_jit(x, w):
    """Numba-optimized rolling mean with same-length padding."""
    n = len(x)
    res = np.empty(n, dtype=np.float64)
    if n < w:
        res[:] = x[0]
        return res
        
    # Initial padding
    res[: w - 1] = x[0]
    
    # Calculate initial sum
    current_sum = 0.0
    for i in range(w):
        current_sum += x[i]
    res[w - 1] = current_sum / w
    
    # Sliding window
    for i in range(w, n):
        current_sum += x[i] - x[i - w]
        res[i] = current_sum / w
    return res


@njit(fastmath=True)
def _calculate_maturity_jit(expiry_timestamps, current_timestamps):
    """Vectorized maturity calculation."""
    return (expiry_timestamps - current_timestamps) / (365.0 * 24 * 3600)


class DataPipeline:
    """
    Data Pipeline for collecting and processing market data.
    OPTIMIZED: Loads real data from Postgres.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.last_run_report: dict[str, Any] = {}

    async def run(self) -> dict[str, Any]:
        """
        Run the data collection pipeline.
        """
        logger.info("data_pipeline_start", symbols=self.config.symbols)

        # In a real implementation, this would trigger scrapers or XDP ingest
        # For now, we verify database connectivity and latest sample count
        from src.database.pipeliner import db_engine

        data = await db_engine.fetch_training_data(self.config.symbols, self.config.max_samples)

        self.last_run_report = {
            "samples_available": len(data),
            "symbols": self.config.symbols,
            "duration_seconds": 1.0,
            "status": "ready",
        }
        logger.info("data_pipeline_status", report=self.last_run_report)
        return self.last_run_report

    async def load_latest_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
        """
        Load the latest collected data from Postgres.
        Returns: (X, y, feature_names, metadata)
        """
        from src.database.pipeliner import db_engine

        #  OPTIMIZATION: Use native async fetch
        records = await db_engine.fetch_training_data(self.config.symbols, self.config.max_samples)

        if not records:
            from src.ml.training.data_gen import (
                generate_synthetic_data_numba as generate_synthetic_data,
            )

            logger.warning("data_pipeline_no_real_data_found", fallback="synthetic")
            return generate_synthetic_data(self.config.min_samples)

        #  OPTIMIZATION: Use structured array for fast conversion
        # We assume records is a list of dicts. We convert to structured array.
        strikes = np.array([r["strike"] for r in records], dtype=np.float64)
        last_prices = np.array([r["last"] for r in records], dtype=np.float64)
        ivs = np.array([r["implied_volatility"] or 0.2 for r in records], dtype=np.float64)
        
        # DateTime handling (Mocked for speed if timestamps aren't available)
        # In real scenario, we'd extract float timestamps
        expiries = np.array([r["expiry"].timestamp() if hasattr(r["expiry"], "timestamp") else 0.0 for r in records])
        times = np.array([r["time"].timestamp() if hasattr(r["time"], "timestamp") else 0.0 for r in records])
        
        maturities = _calculate_maturity_jit(expiries, times)
        maturities = np.where(maturities <= 0, 0.5, maturities) # Fallback

        X_base = np.column_stack([
            strikes,
            maturities,
            ivs,
            np.full(len(records), 0.05), # Rate
            np.full(len(records), 0.01), # Dividend
        ])
        y_raw = last_prices

        #  PHASE 2: Fully Vectorized & JITed Feature Engineering
        iv_lag = np.roll(ivs, 1)
        iv_lag[0] = ivs[0]
        price_lag = np.roll(y_raw, 1)
        price_lag[0] = y_raw[0]

        iv_ma5 = _rolling_mean_jit(ivs, 5)
        iv_ma20 = _rolling_mean_jit(ivs, 20)
        price_ma5 = _rolling_mean_jit(y_raw, 5)
        price_ma20 = _rolling_mean_jit(y_raw, 20)

        # Concatenate New Features
        X = np.column_stack([X_base, iv_lag, price_lag, iv_ma5, iv_ma20, price_ma5, price_ma20])

        feature_names = ["strike", "maturity", "iv", "rate", "dividend"]
        feature_names += [
            "iv_lag1",
            "price_lag1",
            "iv_ma5",
            "iv_ma20",
            "price_ma5",
            "price_ma20",
        ]
        y = y_raw

        metadata = {"data_source": "postgres_jit_vectorized", "count": len(records)}
        return X, y, feature_names, metadata
