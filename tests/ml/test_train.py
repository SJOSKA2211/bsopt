import os
import pytest
from unittest.mock import MagicMock, patch, ANY, PropertyMock
import numpy as np
from src.ml.reinforcement_learning.train import train_td3

@pytest.fixture
def mock_mlflow():
    with patch("src.ml.reinforcement_learning.train.mlflow") as mock:
        mock.start_run.return_value.__enter__.return_value = MagicMock()
        yield mock

@pytest.fixture
def mock_td3():
    with patch("src.ml.reinforcement_learning.train.TD3") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock

def test_train_td3_initialization(mock_mlflow, mock_td3):
    """Test that train_td3 initializes environment, model and MLflow correctly."""
    timesteps = 100
    model_path = "tmp/test_model"
    
    model = train_td3(total_timesteps=timesteps, model_path=model_path)
    
    # Verify MLflow experiment and run
    mock_mlflow.set_experiment.assert_called_with("RL_Trading_Agent")
    mock_mlflow.start_run.assert_called_once()
    
    # Verify model creation
    mock_td3.assert_called_once()
    args, kwargs = mock_td3.call_args
    assert args[0] == "MlpPolicy"
    assert kwargs["verbose"] == 1
    
    # Verify training call
    mock_td3.return_value.learn.assert_called_once_with(
        total_timesteps=timesteps, 
        callback=ANY
    )
    
    # Verify model saving
    mock_td3.return_value.save.assert_called_with(model_path)
    
    assert model == mock_td3.return_value

def test_train_td3_logs_params(mock_mlflow, mock_td3):
    """Test that train_td3 logs parameters to MLflow."""
    train_td3(total_timesteps=100)
    
    mock_mlflow.log_params.assert_called_once()
    params = mock_mlflow.log_params.call_args[0][0]
    assert params["algorithm"] == "TD3"
    assert params["total_timesteps"] == 100

def test_train_td3_logs_model(mock_mlflow, mock_td3):
    """Test that train_td3 logs model to MLflow."""
    train_td3(total_timesteps=100)
    
    mock_mlflow.pytorch.log_model.assert_called_once()

def test_main_function():
    """Test the main function of train.py."""
    with patch("src.ml.reinforcement_learning.train.train_td3") as mock_train:
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(timesteps=1000, output="test_model")
            from src.ml.reinforcement_learning.train import main
            main()
            mock_train.assert_called_once_with(total_timesteps=1000, model_path="test_model")

@patch("src.ml.reinforcement_learning.train.mlflow")
def test_mlflow_callback(mock_mlflow_cb):
    """Test the custom MLflow callback directly."""
    from src.ml.reinforcement_learning.train import MLflowMetricsCallback
    callback = MLflowMetricsCallback()
    
    # Mock the logger
    mock_logger = MagicMock()
    mock_logger.get_log_dict.return_value = {
        "train/actor_loss": 0.5,
        "train/critic_loss": 0.1,
        "rollout/ep_rew_mean": 10.0
    }
    
    # In SB3, logger is often set via init_callback or directly in the object
    with patch.object(MLflowMetricsCallback, 'logger', new_callable=PropertyMock) as mock_logger_prop:
        mock_logger_prop.return_value = mock_logger
        
        # Simulate a step
        callback.num_timesteps = 100
        callback._on_step()
        
        # Verify metrics were logged
        mock_mlflow_cb.log_metrics.assert_called_once()
        metrics = mock_mlflow_cb.log_metrics.call_args[0][0]
        assert metrics["train/actor_loss"] == 0.5
        assert metrics["rollout/ep_rew_mean"] == 10.0
        assert mock_mlflow_cb.log_metrics.call_args[1]["step"] == 100