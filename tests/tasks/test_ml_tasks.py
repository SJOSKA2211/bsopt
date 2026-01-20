import pytest
from unittest.mock import MagicMock, patch
from src.tasks.ml_tasks import train_model_task, hyperparameter_search_task

@pytest.fixture
def mock_celery_eager():
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    return celery_app

@pytest.fixture
def mock_ml_funcs():
    with patch("src.tasks.ml_tasks.train") as mock_train, \
         patch("src.tasks.ml_tasks.run_hyperparameter_optimization") as mock_optimize:
        
        async def async_train(*args, **kwargs):
            return {"run_id": "run-1", "metrics": {"r2": 0.9}, "promoted": True}
            
        async def async_optimize(*args, **kwargs):
            return {"best_params": {"lr": 0.1}, "best_r2": 0.95}
            
        mock_train.side_effect = async_train
        mock_optimize.side_effect = async_optimize
        
        yield mock_train, mock_optimize

def test_train_model_task(mock_celery_eager, mock_ml_funcs):
    res = train_model_task.apply(kwargs={"model_type": "xgboost", "hyperparams": {"depth": 5}})
    result = res.get()
    
    assert result["status"] == "completed"
    assert result["run_id"] == "run-1"
    assert result["metrics"]["r2"] == 0.9

def test_train_model_error(mock_celery_eager):
    with patch("src.tasks.ml_tasks.train", side_effect=Exception("Train Fail")):
        res = train_model_task.apply(kwargs={"model_type": "xgboost"})
        result = res.get()
        assert result["status"] == "failed"
        assert "Train Fail" in result["error"]

def test_hyperparameter_search_task(mock_celery_eager, mock_ml_funcs):
    res = hyperparameter_search_task.apply(kwargs={"model_type": "xgboost", "n_trials": 5})
    result = res.get()
    
    assert result["status"] == "completed"
    assert result["best_r2"] == 0.95

def test_hyperparameter_search_error(mock_celery_eager):
    with patch("src.tasks.ml_tasks.run_hyperparameter_optimization", side_effect=Exception("Optuna Fail")):
        res = hyperparameter_search_task.apply(kwargs={"model_type": "xgboost"})
        result = res.get()
        assert result["status"] == "failed"
        assert "Optuna Fail" in result["error"]
