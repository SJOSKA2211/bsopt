import pytest
import numpy as np
from src.ml.utils.validation import WalkForwardValidator

def test_walk_forward_split():
    """Test that validator provides non-overlapping temporal splits."""
    # 🚀 SINGULARITY: Larger dataset to satisfy TimeSeriesSplit requirements
    X = np.arange(1000).reshape(1000, 1)
    validator = WalkForwardValidator(n_splits=5)
    
    splits = list(validator.split(X))
    assert len(splits) == 5
    
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Ensure past -> future
        assert train_idx[-1] < test_idx[0]
        # Ensure no overlap
        assert set(train_idx).isdisjoint(set(test_idx))

def test_walk_forward_n_splits():
    validator = WalkForwardValidator(n_splits=3)
    assert validator.get_n_splits() == 3
