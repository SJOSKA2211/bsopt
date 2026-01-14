import pytest
from unittest.mock import patch, MagicMock
from src.tasks.ml_tasks import train_model_task, hyperparameter_search_task

@patch("src.tasks.ml_tasks.train")
def test_train_model_task_success(mock_train):
    mock_train.return_value = {
        "run_id": "mlflow-run-123",
        "metrics": {"r2": 0.98},
        "promoted": True
    }
    
    # Try calling without mock_self if __wrapped__ doesn't want it
    result = train_model_task.__wrapped__("xgboost")
    
    assert result["status"] == "completed"
    assert result["run_id"] == "mlflow-run-123"

@patch("src.tasks.ml_tasks.train")
def test_train_model_task_failure(mock_train):
    mock_train.side_effect = Exception("Training failed")
    
    result = train_model_task.__wrapped__("xgboost")
    
    assert result["status"] == "failed"
    assert "Training failed" in result["error"]

@patch("src.tasks.ml_tasks.run_hyperparameter_optimization")
def test_hyperparameter_search_task_success(mock_opt):
    mock_opt.return_value = {
        "best_params": {"learning_rate": 0.1},
        "best_r2": 0.99
    }
    
    result = hyperparameter_search_task.__wrapped__("xgboost", 5)
    
    assert result["status"] == "completed"
    assert result["best_params"]["learning_rate"] == 0.1
