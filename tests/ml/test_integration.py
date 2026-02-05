from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ml.autonomous_pipeline import AutonomousMLPipeline


@pytest.fixture
def mock_config():
    return {
        "ticker": "AAPL",
        "api_key": "test_key",
        "db_url": "sqlite:///:memory:",
        "study_name": "test_pipeline_study",
        "n_trials": 2,
        "framework": "xgboost"
    }

@patch("src.ml.autonomous_pipeline.MarketDataScraper")
@patch("src.ml.autonomous_pipeline.get_db_session")
@patch("src.ml.autonomous_pipeline.InstrumentedTrainer")
@patch("src.ml.autonomous_pipeline.calculate_ks_test")
@patch("src.ml.autonomous_pipeline.PerformanceDriftMonitor")
@patch("src.ml.autonomous_pipeline.setup_logging")
def test_pipeline_run(mock_setup_logging, mock_drift_monitor_class, mock_ks, mock_trainer_class, mock_get_db, mock_scraper_class, mock_config):
    # Mock Scraper
    mock_scraper = mock_scraper_class.return_value
    mock_scraper.fetch_historical_data.return_value = pd.DataFrame({
        "timestamp": [1, 2, 3],
        "open": [100, 101, 102],
        "high": [105, 106, 107],
        "low": [95, 96, 97],
        "close": [102, 103, 104],
        "volume": [1000, 1100, 1200]
    })
    
    # Mock DB Session
    mock_session = mock_get_db.return_value
    
    # Mock Trainer
    mock_trainer = mock_trainer_class.return_value
    mock_study = MagicMock()
    mock_study.best_value = 0.85
    mock_trainer.optimize.return_value = mock_study
    
    # Mock KS
    mock_ks.return_value = (0.1, 0.9)
    
    # Mock Drift Monitor
    mock_monitor = mock_drift_monitor_class.return_value
    mock_monitor.detect_drift.return_value = False
    
    pipeline = AutonomousMLPipeline(mock_config)
    pipeline.run()
    
    assert mock_setup_logging.called
    assert mock_scraper.fetch_historical_data.called
    assert mock_session.add.called
    assert mock_session.commit.called
    assert mock_trainer.optimize.called
    assert mock_ks.called
    assert mock_monitor.add_metric.called
    assert mock_monitor.detect_drift.called