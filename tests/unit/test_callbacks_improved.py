import pytest

from src.ml.callbacks import EarlyStopping


class TestCallbacks:
    def test_early_stopping(self):
        cb = EarlyStopping(patience=3, min_delta=0.01)

        # Improve
        cb(0.9)
        assert cb.best_loss == 0.9
        assert not cb.early_stop

        cb(0.8)
        assert cb.best_loss == 0.8

        # Stagnate
        cb(0.8)
        assert cb.counter == 1
        cb(0.8)
        assert cb.counter == 2
        cb(0.8)
        assert cb.early_stop

