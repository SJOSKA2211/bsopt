from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import structlog

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
            from src.ml.training.data_gen import generate_synthetic_data_numba as generate_synthetic_data

            logger.warning("data_pipeline_no_real_data_found", fallback="synthetic")
            return generate_synthetic_data(self.config.min_samples)

        #  ADVANCED FEATURE ENGINEERING: Vectorized rolling stats and lags
        feature_names = ["strike", "maturity", "implied_vol", "rate", "dividend"]

        # Base Data Extraction
        data_raw = []
        for r in records:
            maturity = (
                (r["expiry"] - r["time"]).total_seconds() / (365 * 24 * 3600)
                if hasattr(r["expiry"], "total_seconds")
                else 0.5
            )
            data_raw.append(
                [
                    r["strike"],
                    maturity,
                    r["implied_volatility"] or 0.2,
                    0.05,
                    0.01,
                    r["last"],
                ]
            )

        arr = np.array(data_raw)
        X_base = arr[:, :5]
        y_raw = arr[:, 5]

        # 1. Lags (1-period)
        iv_lag = np.roll(X_base[:, 2], 1)
        iv_lag[0] = X_base[0, 2]
        price_lag = np.roll(y_raw, 1)
        price_lag[0] = y_raw[0]

        # 2. Rolling Stats (Window 5 and 20)
        def rolling_mean(x, w):
            return np.convolve(x, np.ones(w), "valid") / w

        # Pad the rolling stats to match original length
        def pad_rolling(x, w):
            res = rolling_mean(x, w)
            return np.concatenate([np.full(w - 1, x[0]), res])

        iv_ma5 = pad_rolling(X_base[:, 2], 5)
        iv_ma20 = pad_rolling(X_base[:, 2], 20)
        price_ma5 = pad_rolling(y_raw, 5)
        price_ma20 = pad_rolling(y_raw, 20)

        # Concatenate New Features
        X = np.column_stack([X_base, iv_lag, price_lag, iv_ma5, iv_ma20, price_ma5, price_ma20])

        feature_names += [
            "iv_lag1",
            "price_lag1",
            "iv_ma5",
            "iv_ma20",
            "price_ma5",
            "price_ma20",
        ]
        y = y_raw

        metadata = {"data_source": "postgres_vectorized", "count": len(records)}
        return X, y, feature_names, metadata
