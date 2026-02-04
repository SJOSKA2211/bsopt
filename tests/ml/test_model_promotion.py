import os
import pytest
from unittest.mock import MagicMock, patch

def test_compare_models_script_exists():
    """Verify that the model comparison script exists."""
    assert os.path.exists("src/ml/evaluation/compare_models.py")

def test_promote_script_exists():
    """Verify that the model promotion script exists."""
    assert os.path.exists("src/ml/registry/promote.py")

@patch("mlflow.register_model")
@patch("mlflow.tracking.MlflowClient")
def test_model_promotion_logic(mock_client_class, mock_register):
    """
    Verify the high-level logic for model promotion.
    This is a structural test to ensure the expected components are in place.
    """
    from src.ml.registry.promote import promote_model
    
    # Mock MLflow client behavior
    client_instance = mock_client_class.return_value
    
    # Mock register_model to return a dummy version object
    mock_version = MagicMock()
    mock_version.version = "1"
    mock_register.return_value = mock_version
    
    # Run promotion logic
    promote_model("test-model", "run-123", "production")
    
    # Verify client was called to transition version
    assert client_instance.transition_model_version_stage.called
    args, kwargs = client_instance.transition_model_version_stage.call_args
    assert kwargs['name'] == "test-model"
    assert kwargs['stage'] == "production"