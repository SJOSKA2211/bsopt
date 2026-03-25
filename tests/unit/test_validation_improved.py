import pytest

import numpy as np

from src.ml.utils.validation import WalkForwardValidator

class TestValidation:
    def test_expanding_window(self):
        X = np.arange(100)
        validator = WalkForwardValidator(n_splits=4)
        splits = list(validator.split(X))
        assert len(splits) == 4

        # Check first split
        train, test = splits[0]
        self.assertLess(train[-1], test[0])

        # Check last split
        train, test = splits[-1]
        assert test[-1] == 99

    def test_sliding_window(self):
        X = np.arange(100)
        validator = WalkForwardValidator(n_splits=3, test_size=10)
        splits = list(validator.split(X))
        assert len(splits) == 3

        # In sliding window, test size should be 10
        train, test = splits[0]
        assert len(test) == 10
