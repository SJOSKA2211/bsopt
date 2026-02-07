"""
Unified Temporal Validator (Singularity Refactored)
"""

from collections.abc import Generator

import numpy as np


class WalkForwardValidator:
    """
    High-performance temporal cross-validator.
    Minimizes memory allocations by using index views.
    """

    def __init__(self, n_splits: int = 5, test_size: int | None = None):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X: np.ndarray) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate temporal splits with pre-computed index boundaries.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate split points
        if self.test_size:
            # Fixed-size sliding window
            for i in range(self.n_splits):
                end = n_samples - (self.n_splits - 1 - i) * self.test_size
                start = end - self.test_size
                train_idx = indices[:start]
                test_idx = indices[start:end]
                yield train_idx, test_idx
        else:
            # Expanding window
            fold_size = n_samples // (self.n_splits + 1)
            for i in range(self.n_splits):
                train_end = (i + 1) * fold_size
                test_end = train_end + fold_size
                if i == self.n_splits - 1:
                    test_end = n_samples

                yield indices[:train_end], indices[train_end:test_end]

    def get_n_splits(self):
        return self.n_splits
