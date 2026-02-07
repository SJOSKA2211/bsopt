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
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.last_run_report: dict[str, Any] = {}

    async def run(self) -> dict[str, Any]:
        """
        Run the data collection pipeline.
        """
        logger.info("data_pipeline_start", symbols=self.config.symbols)

        # In a real implementation, this would call scrapers
        # For now, we simulate data collection
        self.last_run_report = {
            "samples_collected": self.config.min_samples,
            "samples_valid": self.config.min_samples,
            "output_path": self.config.output_dir,
            "duration_seconds": 5.0,
            "validation_rate": 1.0,
        }
        logger.info("data_pipeline_completed", report=self.last_run_report)
        return self.last_run_report

    def load_latest_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
        """
        Load the latest collected data.
        Returns: (X, y, feature_names, metadata)
        """
        from src.ml.training.train import generate_synthetic_data

        X, y, features = generate_synthetic_data(self.config.min_samples)
        metadata = {"data_source": "synthetic_pipeline"}
        return X, y, features, metadata
