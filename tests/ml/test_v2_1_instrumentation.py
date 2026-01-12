import numpy as np
from unittest.mock import MagicMock, patch
from prometheus_client import Histogram
from src.shared import observability
from src.ml.trainer import InstrumentedTrainer

def test_training_duration_is_histogram():
    """Verify that ml_training_duration_seconds is a Histogram as per PRD."""
    # Note: In the current code it might be a Summary. This test will fail if it's not a Histogram.
    # The PRD requires: TRAINING_DURATION = Histogram('ml_training_duration_seconds', ...)
    metric = observability.TRAINING_DURATION
    assert isinstance(metric, Histogram), "TRAINING_DURATION should be a Histogram"
    assert metric._name == 'ml_training_duration_seconds'

def test_push_metrics_integration():
    """Verify push_metrics uses the correct gateway URL and job name."""
    with patch("src.shared.observability.push_to_gateway") as mock_push, \
         patch.dict("os.environ", {"PUSHGATEWAY_URL": "http://pushgateway:9091"}):
        
        observability.push_metrics(job_name="test_job")
        
        mock_push.assert_called_once()
        args, kwargs = mock_push.call_args
        assert args[0] == "http://pushgateway:9091"
        assert kwargs['job'] == "test_job"

def test_metric_names_match_prd():
    """Verify metric names match PRD v2.1 spec."""
    assert observability.MODEL_RMSE._name == 'ml_model_rmse'
    assert observability.DATA_DRIFT_SCORE._name == 'ml_data_drift_score'
    # Counter strips _total suffix internally but exposes it.
    assert observability.TRAINING_ERRORS._name == 'ml_training_errors'

def test_trainer_updates_rmse():
    """Verify that trainer updates ml_model_rmse Gauge."""
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    trainer = InstrumentedTrainer(study_name="rmse_test")
    
    with patch("src.ml.trainer.MODEL_RMSE") as mock_rmse:
        mock_labels = MagicMock()
        mock_rmse.labels.return_value = mock_labels
        
        params = {"framework": "xgboost", "max_depth": 3}
        trainer.train_and_evaluate(X, y, params)
        
        mock_rmse.labels.assert_called_with(model_type="xgboost", dataset="validation")
        assert mock_labels.set.called

def test_pipeline_updates_drift_score():
    """Verify that pipeline updates ml_data_drift_score Gauge."""
    config = {
        "api_key": "DEMO_KEY",
        "provider": "mock",
        "db_url": "sqlite:///:memory:",
        "ticker": "AAPL",
        "study_name": "drift_test",
        "n_trials": 1
    }
    
    from src.ml.autonomous_pipeline import AutonomousMLPipeline
    pipeline = AutonomousMLPipeline(config)
    
    # Mock dependencies to reach drift check
    with patch("src.ml.autonomous_pipeline.MarketDataScraper") as mock_scraper_cls, \
         patch("src.ml.autonomous_pipeline.get_db_session"), \
         patch("src.ml.autonomous_pipeline.create_engine"), \
         patch("src.ml.autonomous_pipeline.Base.metadata.create_all"), \
         patch("src.ml.autonomous_pipeline.InstrumentedTrainer"), \
         patch("src.ml.autonomous_pipeline.DATA_DRIFT_SCORE") as mock_drift_gauge, \
         patch("src.ml.autonomous_pipeline.KS_TEST_SCORE") as mock_ks_gauge:
        
        mock_scraper = mock_scraper_cls.return_value
        import pandas as pd
        mock_scraper.fetch_historical_data.return_value = pd.DataFrame({
            "timestamp": [1672531200000] * 10,
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000] * 10
        })
        
        try:
            pipeline.run()
        except Exception:
            pass # We care about the drift score update
            
        assert mock_drift_gauge.set.called
        assert mock_ks_gauge.set.called

def test_structlog_json_formatting():
    """Verify that structlog is configured with JSONRenderer."""
    import structlog
    from src.shared.observability import setup_logging
    
    # Reset structlog config to test our setup
    structlog.reset_defaults()
    setup_logging()
    
    config = structlog.get_config()
    processors = config['processors']
    
    # Check if JSONRenderer is in the processor list
    from structlog.processors import JSONRenderer
    assert any(isinstance(p, JSONRenderer) for p in processors)


