import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Generator, Tuple, Optional

class WalkForwardValidator:
    """
    🚀 SOTA: Unified Temporal Cross-Validator.
    Provides expanding and sliding window splits for robust time-series validation.
    """
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate temporal train/test indices.
        Ensures that test_index always follows train_index chronologically.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        for train_index, test_index in tscv.split(X):
            # 🚀 SINGULARITY: Validation check for no-leakage
            if len(train_index) > 0 and len(test_index) > 0:
                if train_index[-1] >= test_index[0]:
                    raise ValueError("Temporal leakage detected: train index overlaps with test index.")
            yield train_index, test_index

    def get_n_splits(self):
        return self.n_splits
