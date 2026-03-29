from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ml.trainer import ModelTrainer, PyTorchTrainer

@pytest.fixture
def dummy_data():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)  # Regression targets
    return X, y

@pytest.fixture
def mock_tracker():
    with patch("src.ml.trainer.ExperimentTracker") as mock:
        tracker = MagicMock()
        mock.return_value = tracker
        tracker.start_run.return_value.__enter__.return_value = MagicMock()
        yield tracker

def test_xgboost_trainer(mock_tracker, dummy_data):
    X, y = dummy_data
    trainer = ModelTrainer(study_name="test_xgboost")
    params = {"framework": "xgboost", "n_estimators": 10, "max_depth": 3}

    with patch("src.ml.strategies.XGBoostStrategy.train") as mock_train:
        mock_train.return_value = MagicMock()
        # Mock predict to return something for regression metrics
        with patch("src.ml.strategies.XGBoostStrategy.predict") as mock_pred:
            mock_pred.return_value = np.random.rand(len(y) // 5)  # 20% test size
            r2 = trainer.train_and_evaluate(X, y, params)
            assert isinstance(r2, float)
            mock_tracker.log_metrics.assert_called()

def test_sklearn_trainer(mock_tracker, dummy_data):
    X, y = dummy_data
    trainer = ModelTrainer(study_name="test_sklearn")
    params = {"framework": "sklearn", "n_estimators": 5}

    with patch("src.ml.strategies.SklearnStrategy.train") as mock_train:
        mock_train.return_value = MagicMock()
        with patch("src.ml.strategies.SklearnStrategy.predict") as mock_pred:
            mock_pred.return_value = np.random.rand(len(y) // 5)
            r2 = trainer.train_and_evaluate(X, y, params)
            assert isinstance(r2, float)

def test_pytorch_trainer(mock_tracker, dummy_data):
    X, y = dummy_data
    trainer = ModelTrainer(study_name="test_pytorch")
    params = {"framework": "pytorch", "epochs": 2, "lr": 0.01}

    with patch("src.ml.strategies.PyTorchStrategy.train") as mock_train:
        mock_train.return_value = MagicMock()
        with patch("src.ml.strategies.PyTorchStrategy.predict") as mock_pred:
            mock_pred.return_value = np.random.rand(len(y) // 5)
            r2 = trainer.train_and_evaluate(X, y, params)
            assert isinstance(r2, float)

def test_optimize(mock_tracker):
    with patch("src.ml.trainer.optuna.create_study") as mock_create:
        mock_study = MagicMock()
        mock_study.best_params = {"a": 1}
        mock_study.best_value = 0.9
        mock_create.return_value = mock_study

        trainer = ModelTrainer(study_name="test_opt")
        objective = MagicMock()
        study = trainer.optimize(objective, n_trials=1)

        assert study == mock_study
        assert trainer.best_params == {"a": 1}

def test_pytorch_wrapper(mock_tracker, dummy_data):
    X, y = dummy_data
    trainer = PyTorchTrainer(study_name="test_wrapper")
    with patch.object(ModelTrainer, "train_and_evaluate", return_value=0.9) as mock_train:
        res = trainer.train(X, y, {"epochs": 1})
        assert res == 0.9
        mock_train.assert_called_once()

def test_push_metrics_alias(mock_tracker):
    trainer = ModelTrainer(study_name="test_alias")
    trainer.push_metrics()
    mock_tracker.push_to_gateway.assert_called_once()
