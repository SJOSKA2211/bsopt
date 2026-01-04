import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.ml.trainer import InstrumentedTrainer

@pytest.fixture
def sample_data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_trainer_initialization():
    trainer = InstrumentedTrainer(study_name="test_study")
    assert trainer.study_name == "test_study"
    assert trainer.model is None

@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metric")
def test_trainer_train_xgboost(mock_log_metric, mock_log_params, mock_start_run, sample_data):
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="test_study")
    
    # Define a simple search space for Optuna
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 20),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1)
        }
        return trainer.train_and_evaluate(X, y, params)

    study = trainer.optimize(objective, n_trials=2)
    
    assert len(study.trials) == 2
    assert "n_estimators" in study.best_params
    assert trainer.model is not None
    assert mock_start_run.called
    assert mock_log_params.called
    assert mock_log_metric.called

def test_trainer_prometheus_metrics(sample_data):
    # This might require checking if prometheus metrics are registered
    # For now, let's just ensure the trainer has the metrics defined
    trainer = InstrumentedTrainer(study_name="test_study")
    assert hasattr(trainer, "training_duration_metric")
    assert hasattr(trainer, "model_accuracy_metric")
