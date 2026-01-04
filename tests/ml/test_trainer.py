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

@patch("src.ml.trainer.logger")
def test_trainer_logging(mock_logger, sample_data):
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="test_study")
    params = {"max_depth": 3, "learning_rate": 0.1}
    
    trainer.train_and_evaluate(X, y, params)
    
    assert mock_logger.info.called

def test_trainer_prometheus_metrics_usage(sample_data):
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="test_study")
    
    # Mock the metrics to verify they are called
    trainer.training_duration_metric = MagicMock()
    trainer.model_accuracy_metric = MagicMock()
    
    params = {"max_depth": 3, "learning_rate": 0.1}
    trainer.train_and_evaluate(X, y, params)
    
    trainer.training_duration_metric.observe.assert_called_once()
    trainer.model_accuracy_metric.set.assert_called_once()

@patch("mlflow.set_tracking_uri")
def test_trainer_mlflow_uri(mock_set_uri):
    trainer = InstrumentedTrainer(study_name="test_study", tracking_uri="http://localhost:5000")
    mock_set_uri.assert_called_once_with("http://localhost:5000")

@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metric")
def test_mlflow_logging_content(mock_log_metric, mock_log_params, mock_start_run, sample_data):
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="test_study")
    params = {"max_depth": 3, "learning_rate": 0.1}
    
    trainer.train_and_evaluate(X, y, params)
    
    mock_log_params.assert_called_with(params)
    # Check if accuracy was logged
    metric_names = [call.args[0] for call in mock_log_metric.call_args_list]
    assert "accuracy" in metric_names
    assert "duration" in metric_names
