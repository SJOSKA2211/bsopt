from unittest.mock import MagicMock, call, patch

import pytest

from src.aiops.ml_pipeline_trigger import MLPipelineTrigger  # Assuming this path


@patch("src.aiops.ml_pipeline_trigger.logger")
@patch("src.aiops.ml_pipeline_trigger.AutonomousMLPipeline")
class TestMLPipelineTrigger:
    def test_ml_pipeline_trigger_init_success(self, mock_pipeline_class, mock_logger):
        """Test successful initialization of MLPipelineTrigger."""
        config = {"ticker": "AAPL", "framework": "xgboost"}
        trigger = MLPipelineTrigger(config)
        assert trigger.config == config
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_ml_pipeline_trigger_missing_config_raises_error(self, mock_pipeline_class, mock_logger):
        """Test that missing config parameters raise an error."""
        # Missing 'ticker'
        with pytest.raises(ValueError, match="ML Pipeline config must contain 'ticker' and 'framework'"):
            MLPipelineTrigger({"framework": "xgboost"})
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_pipeline_class.assert_not_called()
        
        # Missing 'framework'
        with pytest.raises(ValueError, match="ML Pipeline config must contain 'ticker' and 'framework'"):
            MLPipelineTrigger({"ticker": "AAPL"})
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_pipeline_class.assert_not_called()


    def test_ml_pipeline_trigger_success(self, mock_pipeline_class, mock_logger):
        """Test successful triggering of the ML pipeline."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        config = {"key": "value", "ticker": "AAPL", "framework": "xgboost"}
        trigger = MLPipelineTrigger(config)
        mock_logger.reset_mock() # Reset logger calls after init

        result = trigger.trigger_retraining()
        
        mock_pipeline_class.assert_called_once_with(config)
        mock_pipeline_instance.run.assert_called_once()
        assert result
        mock_logger.info.assert_has_calls([
            call("ml_pipeline_trigger", status="attempting_retraining", config=config),
            call("ml_pipeline_trigger", status="success", message="ML retraining pipeline triggered successfully.")
        ])
        mock_logger.error.assert_not_called()

    def test_ml_pipeline_trigger_failure(self, mock_pipeline_class, mock_logger):
        """Test triggering of the ML pipeline when it fails."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.side_effect = Exception("Pipeline failed")
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        config = {"key": "value", "ticker": "AAPL", "framework": "xgboost"}
        trigger = MLPipelineTrigger(config)
        mock_logger.reset_mock() # Reset logger calls after init
        
        result = trigger.trigger_retraining()
        
        mock_pipeline_class.assert_called_once_with(config)
        mock_pipeline_instance.run.assert_called_once()
        assert not result
        mock_logger.info.assert_called_once_with("ml_pipeline_trigger", status="attempting_retraining", config=config)
        mock_logger.error.assert_called_once_with(
            "ml_pipeline_trigger", status="failure", error="Pipeline failed", message="Failed to trigger ML retraining pipeline."
        )