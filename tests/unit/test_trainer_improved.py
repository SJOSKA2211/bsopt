import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys

# Mock dependencies before imports
sys.modules["mlflow"] = MagicMock()
sys.modules["optuna"] = MagicMock()
sys.modules["optuna.exceptions"] = MagicMock()
sys.modules["optuna.pruners"] = MagicMock()

from src.ml.trainer import ModelTrainer

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(100, 5)
        self.y = np.random.rand(100)
        self.params = {"framework": "sklearn", "n_estimators": 10}
        
        with (
            patch("src.ml.trainer.ExperimentTracker"),
            patch("src.ml.trainer.ModelQuantizer")
        ):
            self.trainer = ModelTrainer(study_name="test_study")

    @patch("src.ml.trainer.get_strategy")
    @patch("src.ml.trainer.calculate_regression_metrics")
    def test_train_and_evaluate(self, mock_metrics, mock_get_strategy):
        mock_strategy = MagicMock()
        mock_strategy.train.return_value = MagicMock()
        mock_strategy.predict.return_value = np.random.rand(20)
        mock_get_strategy.return_value = mock_strategy
        
        mock_metrics.return_value = {"mae": 0.1, "rmse": 0.2, "r2": 0.9}
        
        r2 = self.trainer.train_and_evaluate(self.X, self.y, self.params)
        self.assertEqual(r2, 0.9)
        self.assertTrue(mock_strategy.train.called)

    @patch("src.ml.trainer.optuna.create_study")
    def test_optimize(self, mock_create_study):
        mock_study = MagicMock()
        mock_study.best_params = {"max_depth": 5}
        mock_study.best_value = 0.95
        mock_create_study.return_value = mock_study
        
        objective = MagicMock()
        study = self.trainer.optimize(objective, n_trials=1)
        
        self.assertEqual(study.best_value, 0.95)
        self.assertTrue(mock_study.optimize.called)

if __name__ == '__main__':
    unittest.main()
