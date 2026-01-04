import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ml.trainer import InstrumentedTrainer
from src.shared import observability

@pytest.fixture
def sample_data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture(autouse=True)
def mock_mlflow():
    with patch("mlflow.start_run"), \
         patch("mlflow.log_params"), \
         patch("mlflow.log_metric"), \
         patch("mlflow.set_tracking_uri"), \
         patch("mlflow.log_artifact"), \
         patch("mlflow.xgboost.log_model"), \
         patch("mlflow.sklearn.log_model"), \
         patch("mlflow.pytorch.log_model"):
        yield

def test_trainer_new_prometheus_metrics(sample_data):
    """Verify that new Prometheus metrics (RMSE, Errors, duration Histogram) are updated."""
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="metrics_test")
    
    # Patch where they are used (src.ml.trainer)
    with patch("src.ml.trainer.TRAINING_DURATION_HISTOGRAM") as mock_hist, \
         patch("src.ml.trainer.MODEL_RMSE") as mock_rmse:
        
        mock_hist_labels = MagicMock()
        mock_hist.labels.return_value = mock_hist_labels
        
        mock_rmse_labels = MagicMock()
        mock_rmse.labels.return_value = mock_rmse_labels
        
        params = {"max_depth": 3, "learning_rate": 0.1, "framework": "xgboost"}
        trainer.train_and_evaluate(X, y, params)
        
        # Verify Histogram
        mock_hist.labels.assert_called_with(framework="xgboost")
        assert mock_hist_labels.observe.called
        
        # Verify RMSE Gauge
        mock_rmse.labels.assert_called_with(model_type="xgboost", dataset="validation")
        assert mock_rmse_labels.set.called

def test_trainer_error_metrics(sample_data):
    """Verify that training errors increment the error counter."""
    X, y = sample_data
    trainer = InstrumentedTrainer(study_name="error_test")
    
    with patch("src.ml.trainer.TRAINING_ERRORS") as mock_errors, \
         patch("src.ml.trainer.train_test_split", side_effect=Exception("Simulated crash")):
        
        mock_errors_labels = MagicMock()
        mock_errors.labels.return_value = mock_errors_labels
        
        params = {"framework": "xgboost"}
        
        with pytest.raises(Exception, match="Simulated crash"):
            trainer.train_and_evaluate(X, y, params)
            
        mock_errors.labels.assert_called_with(framework="xgboost")
        assert mock_errors_labels.inc.called

def test_structlog_configuration():
    """Verify that structlog is configured correctly."""
    import structlog
    from src.shared.observability import setup_logging
    
    setup_logging()
    
    # Check if JSONRenderer is in processors
    config = structlog.get_config()
    processors = config['processors']
    
    renderer_types = [type(p) for p in processors]
    assert any("JSONRenderer" in str(t) for t in renderer_types)
    assert any("TimeStamper" in str(t) for t in renderer_types)
