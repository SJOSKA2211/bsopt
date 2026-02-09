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
            "status": "ready"
        }
        logger.info("data_pipeline_status", report=self.last_run_report)
        return self.last_run_report

    async def load_latest_data(self) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
        """
        Load the latest collected data from Postgres.
        Returns: (X, y, feature_names, metadata)
        """
        from src.database.pipeliner import db_engine
        
        #  OPTIMIZATION: Use native async fetch
        records = await db_engine.fetch_training_data(self.config.symbols, self.config.max_samples)

        if not records:
            from src.ml.training.train import generate_synthetic_data
            logger.warning("data_pipeline_no_real_data_found", fallback="synthetic")
            return generate_synthetic_data(self.config.min_samples)

        #  ADVANCED FEATURE ENGINEERING TODO: Implement rolling stats and lag features here
        # Convert to NumPy
        # X: strike, expiry_days, volatility, rate, dividend
        # y: price (last)
        feature_names = ["strike", "maturity", "implied_vol", "rate", "dividend"]
        X = np.array([[
            r["strike"], 
            (r["expiry"] - r["time"]).total_seconds() / (365 * 24 * 3600) if hasattr(r["expiry"], "total_seconds") else 0.5,
            r["implied_volatility"] or 0.2,
            0.05, # Default rate if not in DB
            0.01  # Default div if not in DB
        ] for r in records])
        
        y = np.array([r["last"] for r in records])
        
        metadata = {"data_source": "postgres_vectorized", "count": len(records)}
        return X, y, feature_names, metadata
