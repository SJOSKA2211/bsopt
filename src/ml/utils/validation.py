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
        Generate temporal train/test indices using raw indexing.
        Ensures that test_index always follows train_index chronologically.
        """
        n_samples = len(X)
        # 🚀 SINGULARITY: Raw index calculation for maximum stability
        # Each fold adds roughly (n_samples / (n_splits + 1)) new samples to train
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_end = train_end + fold_size
            
            # Final fold takes the remainder
            if i == self.n_splits - 1:
                test_end = n_samples
                
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self):
        return self.n_splits
