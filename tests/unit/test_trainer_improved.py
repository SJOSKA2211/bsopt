import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock dependencies before imports
sys.modules["mlflow"] = MagicMock()
sys.modules["optuna"] = MagicMock()
sys.modules["optuna.exceptions"] = MagicMock()
sys.modules["optuna.pruners"] = MagicMock()

from src.ml.trainer import ModelTrainer


@pytest.fixture
def trainer_data():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    params = {"framework": "sklearn", "n_estimators": 10}
    return X, y, params

@pytest.fixture
def trainer():
    with (
        patch("src.ml.trainer.ExperimentTracker"),
        patch("src.ml.trainer.ModelQuantizer"),
    ):
        return ModelTrainer(study_name="test_study")

@patch("src.ml.trainer.get_strategy")
@patch("src.ml.trainer.calculate_regression_metrics")
def test_train_and_evaluate(mock_metrics, mock_get_strategy, trainer, trainer_data):
    X, y, params = trainer_data
    mock_strategy = MagicMock()
    mock_strategy.train.return_value = MagicMock()
    mock_strategy.predict.return_value = np.random.rand(20)
    mock_get_strategy.return_value = mock_strategy

    mock_metrics.return_value = {"mae": 0.1, "rmse": 0.2, "r2": 0.9}

    r2 = trainer.train_and_evaluate(X, y, params)
    assert r2 == 0.9
    assert mock_strategy.train.called

@patch("src.ml.trainer.optuna.create_study")
def test_optimize(mock_create_study, trainer):
    mock_study = MagicMock()
    mock_study.best_params = {"max_depth": 5}
    mock_study.best_value = 0.95
    mock_create_study.return_value = mock_study

    objective = MagicMock()
    study = trainer.optimize(objective, n_trials=1)

    assert study.best_value == 0.95
    assert mock_study.optimize.called
