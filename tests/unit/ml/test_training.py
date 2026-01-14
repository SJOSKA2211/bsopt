import pytest
import numpy as np
from src.ml.training.train import generate_synthetic_data, train, load_or_collect_data
from unittest.mock import patch, MagicMock

def test_generate_synthetic_data():
    n = 100
    X, y, features = generate_synthetic_data(n)
    assert X.shape == (n, 9)
    assert len(y) == n
    assert len(features) == 9
    # check for 'underlying_price' instead of 'moneyness' as seen in source earlier
    assert "underlying_price" in features

@pytest.mark.asyncio
async def test_load_or_collect_data_synthetic():
    X, y, features, meta = await load_or_collect_data(use_real_data=False, n_samples=50)
    assert meta["data_source"] == "synthetic"
    assert len(y) == 50

@pytest.mark.asyncio
@patch("mlflow.active_run")
@patch("mlflow.set_experiment")
@patch("mlflow.set_tracking_uri")
@patch("mlflow.log_params")
@patch("mlflow.log_metric")
@patch("mlflow.start_run")
@patch("mlflow.xgboost.log_model")
async def test_train_smoke(mock_log_model, mock_start_run, mock_log_params, mock_log_metric, mock_set_uri, mock_set_exp, mock_active_run):
    # Mock MLflow to avoid actual tracking issues in unit tests
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_start_run.return_value.__enter__.return_value = mock_run
    mock_active_run.return_value = mock_run
    
    mock_model_info = MagicMock()
    mock_model_info.registered_model_version = "1"
    mock_log_model.return_value = mock_model_info
    
    # Also mock MlflowClient
    with patch("src.ml.training.train.MlflowClient") as mock_client:
        result = await train(use_real_data=False, n_samples=100, promote_threshold=0.0)
        assert "run_id" in result
        assert result["r2"] is not None