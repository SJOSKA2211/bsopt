import pytest
from unittest.mock import MagicMock, patch
from src.aiops.ml_pipeline_trigger import MLPipelineTrigger # Assuming this path

@patch("src.aiops.ml_pipeline_trigger.AutonomousMLPipeline")
def test_ml_pipeline_trigger_init(mock_pipeline_class):
    """Test initialization of MLPipelineTrigger."""
    config = {"key": "value", "ticker": "AAPL", "framework": "xgboost"}
    trigger = MLPipelineTrigger(config)
    assert trigger.config == config

@patch("src.aiops.ml_pipeline_trigger.AutonomousMLPipeline")
def test_ml_pipeline_trigger_success(mock_pipeline_class):
    """Test successful triggering of the ML pipeline."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline_instance
    
    config = {"key": "value", "ticker": "AAPL", "framework": "xgboost"}
    trigger = MLPipelineTrigger(config)
    
    result = trigger.trigger_retraining()
    
    mock_pipeline_class.assert_called_once_with(config)
    mock_pipeline_instance.run.assert_called_once()
    assert result

@patch("src.aiops.ml_pipeline_trigger.AutonomousMLPipeline")
def test_ml_pipeline_trigger_failure(mock_pipeline_class):
    """Test triggering of the ML pipeline when it fails."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.side_effect = Exception("Pipeline failed")
    mock_pipeline_class.return_value = mock_pipeline_instance
    
    config = {"key": "value", "ticker": "AAPL", "framework": "xgboost"}
    trigger = MLPipelineTrigger(config)
    
    result = trigger.trigger_retraining()
    
    mock_pipeline_class.assert_called_once_with(config)
    mock_pipeline_instance.run.assert_called_once()
    assert not result

def test_ml_pipeline_trigger_missing_config_raises_error():
    """Test that missing config parameters raise an error."""
    # Missing 'ticker'
    with pytest.raises(ValueError, match="ML Pipeline config must contain 'ticker' and 'framework'"):
        MLPipelineTrigger({"framework": "xgboost"})
    
    # Missing 'framework'
    with pytest.raises(ValueError, match="ML Pipeline config must contain 'ticker' and 'framework'"):
        MLPipelineTrigger({"ticker": "AAPL"})
