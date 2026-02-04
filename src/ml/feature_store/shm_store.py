"""
SHM-Backed Feature Store (Singularity Refactored)
==============================================

Reads real-time market data from shared memory for ultra-low latency feature engineering.
"""

import pandas as pd
import structlog
from src.shared.shm_manager import SHMManager
from .base import FeatureStore
from typing import List, Dict, Any

logger = structlog.get_logger(__name__)

class SHMFeatureStore(FeatureStore):
    """
    Feature Store that leverages the 'market_mesh' shared memory segment.
    """
    def __init__(self):
        self.shm_reader = SHMManager("market_mesh", dict)
        self._registry = {}

    def get_latest_market_snapshot(self) -> pd.DataFrame:
        """Reads the latest data from the Market Mesh SHM."""
        try:
            data = self.shm_reader.read()
            # Convert dict of dicts to DataFrame
            # data format: {symbol: {price: 100, volume: 1000, ...}}
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index.name = 'symbol'
            return df
        except Exception as e:
            logger.error("shm_read_failed", error=str(e))
            return pd.DataFrame()

    def compute_features(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Reads latest data and computes requested features.
        """
        df = self.get_latest_market_snapshot()
        if df.empty:
            return df

        # Apply transformations from registry
        for name in feature_names:
            if name in self._registry:
                df[name] = self._registry[name].transform(df)
        
        return df

    def register(self, feature: Any):
        self._registry[feature.name] = feature

# God-Mode instance
shm_feature_store = SHMFeatureStore()
