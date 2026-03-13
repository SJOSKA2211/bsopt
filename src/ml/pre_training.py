import numpy as np
import pandas as pd
from numba import njit
from typing import Any, cast

@njit(cache=True, fastmath=True)
def optimized_normalization_kernel(data: np.ndarray, means: np.ndarray, stds: np.ndarray):
    """Numba kernel for zero-copy feature normalization."""
    return (data - means) / stds

class MLPreTrainer:
    """
    Handles high-performance feature engineering and data preparation
    using vectorized kernels and Numba.
    """
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """
        Processes raw DataFrame into normalized NumPy tensor for PyTorch.
        """
        # 1. Cleaning with precision
        df = df.copy()
        for feat in features:
            if df[feat].isnull().any():
                # Efficient fill
                median_val = df[feat].median()
                df[feat] = df[feat].fillna(median_val)
        
        data_raw = df[features].values.astype(np.float64)
        
        # 2. Optimized Normalization
        means = np.mean(data_raw, axis=0)
        stds = np.std(data_raw, axis=0)
        stds[stds == 0] = 1.0 # Avoid division by zero
        
        normalized_data = optimized_normalization_kernel(data_raw, means, stds)
        
        return normalized_data, means, stds

    @staticmethod
    def calculate_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamically calculates cross-sectional features like sector rank.
        """
        if "sector" in df.columns and "market_cap" in df.columns:
            df["sector_cap_rank"] = df.groupby("sector")["market_cap"].rank(pct=True)
            
        return df
