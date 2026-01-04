import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ml.trainer import InstrumentedTrainer
from src.shared import observability
from src.ml.autonomous_pipeline import AutonomousMLPipeline

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

def test_pipeline_scrape_metrics():
    """Verify that scrape duration and errors are tracked in the pipeline."""
    config = {
        "api_key": "TEST",
        "db_url": "sqlite:///:memory:",
        "ticker": "TEST",
        "study_name": "test_study"
    }
    
    # Test Error Path
    with patch("src.ml.autonomous_pipeline.MarketDataScraper") as mock_scraper_cls, \
         patch("src.ml.autonomous_pipeline.SCRAPE_ERRORS") as mock_errors, \
         patch("src.ml.autonomous_pipeline.get_db_session"), \
         patch("src.ml.autonomous_pipeline.create_engine"), \
         patch("src.ml.autonomous_pipeline.Base.metadata.create_all"), \
         patch("src.ml.autonomous_pipeline.PerformanceDriftMonitor"):
             
        mock_scraper = mock_scraper_cls.return_value
        mock_scraper.fetch_historical_data.side_effect = Exception("API Fail")
        
        mock_errors_labels = MagicMock()
        mock_errors.labels.return_value = mock_errors_labels
        
        pipeline = AutonomousMLPipeline(config)
        
        # We expect a critical failure log but also the counter increment
        with pytest.raises(Exception, match="API Fail"):
            pipeline.run()
            
        mock_errors.labels.assert_called_with(api="alpha_vantage", status_code="error")
        assert mock_errors_labels.inc.called

    # Test Success Path (Duration)
    with patch("src.ml.autonomous_pipeline.MarketDataScraper") as mock_scraper_cls, \
         patch("src.ml.autonomous_pipeline.SCRAPE_DURATION") as mock_duration, \
         patch("src.ml.autonomous_pipeline.get_db_session"), \
         patch("src.ml.autonomous_pipeline.create_engine"), \
         patch("src.ml.autonomous_pipeline.Base.metadata.create_all"), \
         patch("src.ml.autonomous_pipeline.InstrumentedTrainer"), \
         patch("src.ml.autonomous_pipeline.PerformanceDriftMonitor"):
             
        mock_scraper = mock_scraper_cls.return_value
        import pandas as pd
        mock_scraper.fetch_historical_data.return_value = pd.DataFrame({
             "timestamp": ["2023-01-01"], "open": [100], "high": [105], "low": [99], "close": [101], "volume": [1000]
        })
        
        mock_duration_labels = MagicMock()
        mock_duration.labels.return_value = mock_duration_labels
        
        pipeline = AutonomousMLPipeline(config)
        
        try:
             pipeline.run()
        except:
             pass 
             
        mock_duration.labels.assert_called_with(api="alpha_vantage")
        assert mock_duration_labels.time.called