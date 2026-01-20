import pytest
from unittest.mock import MagicMock, patch, ANY, PropertyMock
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
    
    # Mock eval_callback to avoid AttributeError if implementation expects results from it
    with patch("src.ml.reinforcement_learning.train.EvalCallback") as mock_eval:
        mock_eval_instance = MagicMock()
        mock_eval_instance.last_mean_reward = [10.0]
        mock_eval.return_value = mock_eval_instance
        
        result = train_td3(total_timesteps=timesteps, model_path=model_path)
    
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
    
    # New return signature: dict of metadata
    assert isinstance(result, dict)
    assert result["mean_reward"] == 10.0
    assert result["model_path"] == model_path

@patch("src.ml.reinforcement_learning.train.RAY_AVAILABLE", True)
@patch("src.ml.reinforcement_learning.train.ray.init")
@patch("src.ml.reinforcement_learning.train.ray.get")
@patch("src.ml.reinforcement_learning.train.train_td3_remote.remote")
def test_train_distributed_function(mock_train_td3_remote, mock_ray_get, mock_ray_init):
    """Test the train_distributed function."""
    mock_train_td3_remote.return_value = "future_object"
    # New return includes both results and best_result
    mock_results = [{"mean_reward": 10.0, "model_path": "path1"}, {"mean_reward": 20.0, "model_path": "path2"}]
    mock_ray_get.return_value = mock_results
    
    from src.ml.reinforcement_learning.train import train_distributed
    
    num_instances = 2
    total_timesteps = 50
    ray_address = "localhost:6379"
    
    results, best_result = train_distributed(num_instances=num_instances, total_timesteps=total_timesteps, ray_address=ray_address)
    
    mock_ray_init.assert_called_once_with(address=ray_address, ignore_reinit_error=True)
    assert mock_train_td3_remote.call_count == num_instances
    mock_ray_get.assert_called_once_with(["future_object", "future_object"])
    assert results == mock_results
    assert best_result == mock_results[1]

def test_train_td3_logs_params(mock_mlflow, mock_td3):
    """Test that train_td3 logs parameters to MLflow."""
    # Mock eval_callback
    with patch("src.ml.reinforcement_learning.train.EvalCallback") as mock_eval:
        mock_eval.return_value.last_mean_reward = [10.0]
        train_td3(total_timesteps=100)
    
    mock_mlflow.log_params.assert_called_once()
    params = mock_mlflow.log_params.call_args[0][0]
    assert params["algorithm"] == "TD3"
    assert params["total_timesteps"] == 100

def test_train_td3_logs_model(mock_mlflow, mock_td3):
    """Test that train_td3 logs model to MLflow."""
    # Mock eval_callback
    with patch("src.ml.reinforcement_learning.train.EvalCallback") as mock_eval:
        mock_eval.return_value.last_mean_reward = [10.0]
        train_td3(total_timesteps=100)
    
    mock_mlflow.pytorch.log_model.assert_called_once()

def test_main_function():
    """Test the main function of train.py."""
    with patch("src.ml.reinforcement_learning.train.train_td3") as mock_train:
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                timesteps=1000, 
                output="test_model",
                distributed=False,
                instances=2,
                ray_address="auto"
            )
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
    mock_logger.name_to_value = {
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

@patch("src.ml.reinforcement_learning.train.RAY_AVAILABLE", False)
def test_train_distributed_ray_not_available():
    """Test train_distributed when Ray is not available."""
    from src.ml.reinforcement_learning.train import train_distributed
    with patch("src.ml.reinforcement_learning.train.logger.error") as mock_logger_error:
        result = train_distributed(num_instances=1, total_timesteps=1)
        assert result is None
        mock_logger_error.assert_called_once_with(
            "ray_not_available", message="Ray not installed. Cannot run distributed training."
        )

@patch("src.ml.reinforcement_learning.train.train_td3")
@patch("src.ml.reinforcement_learning.train.train_distributed")
@patch("argparse.ArgumentParser.parse_args")
def test_main_function_distributed(mock_args, mock_train_distributed, mock_train_td3):
    """Test the main function for distributed training."""
    mock_args.return_value = MagicMock(
        timesteps=1000, 
        output="test_model", 
        distributed=True,
        instances=3,
        ray_address="auto"
    )
    from src.ml.reinforcement_learning.train import main
    main()
    mock_train_distributed.assert_called_once_with(
        num_instances=3, total_timesteps=1000, ray_address="auto"
    )
    mock_train_td3.assert_not_called()