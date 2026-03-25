import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Pre-emptive mocking
sys.modules["mlflow"] = MagicMock()
sys.modules["optuna"] = MagicMock()
sys.modules["optuna.exceptions"] = MagicMock()
sys.modules["optuna.pruners"] = MagicMock()

from src.ml.trainer import ModelTrainer


class TestModelTrainerPhase5:
    def setUp(self):
        self.X = np.random.rand(100, 5)
        self.y = np.random.rand(100)
        self.params = {"framework": "xgboost", "n_estimators": 10}

        # Mock tracker to avoid MLflow calls
        with (
            patch("src.ml.trainer.ExperimentTracker"),
            patch("src.ml.trainer.ModelQuantizer"),
        ):
            self.trainer = ModelTrainer(study_name="test_study", n_splits=3)

    @patch("src.ml.trainer.get_strategy")
    def test_train_and_evaluate_walk_forward(self, mock_get_strategy):
        """Verify that training uses multiple folds when n_splits > 1."""
        mock_strategy = MagicMock()
        mock_strategy.train.return_value = MagicMock()
        mock_strategy.predict.return_value = np.random.rand(
            25
        )  # Size of test set in expanding window
        mock_get_strategy.return_value = mock_strategy

        # We expect 3 folds
        result = self.trainer.train_and_evaluate(self.X, self.y, self.params)

        # strategy.train should be called 3 times
        assert mock_strategy.train.call_count == 3
        assert isinstance(result, float)

    @patch("src.ml.trainer.ModelScorecard")
    @patch("src.ml.trainer.get_strategy")
    def test_holistic_scorecard_integration(self, mock_get_strategy, mock_scorecard_cls):
        """Verify that ModelScorecard is used for evaluation."""
        mock_strategy = MagicMock()
        mock_get_strategy.return_value = mock_strategy

        mock_scorecard = MagicMock()
        mock_scorecard.to_dict.return_value = {
            "rmse": 0.1,
            "r2": 0.9,
            "mae": 0.05,
            "mape": 0.01,
            "max_pe": 0.02,
            "pricing_bias": 0.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.1,
            "score": 0.8,
        }
        mock_scorecard_cls.return_value = mock_scorecard

        result = self.trainer.train_and_evaluate(self.X, self.y, self.params)

        assert result == 0.9
        mock_scorecard_cls.assert_called()

