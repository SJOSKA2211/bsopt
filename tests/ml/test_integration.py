import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.ml.autonomous_pipeline import AutonomousMLPipeline

@pytest.fixture
def mock_config():
    return {
        "ticker": "AAPL",
        "api_key": "test_key",
        "db_url": "sqlite:///:memory:",
        "study_name": "test_pipeline_study",
        "n_trials": 2
    }

@patch("src.ml.autonomous_pipeline.MarketDataScraper")
@patch("src.ml.autonomous_pipeline.get_db_session")
@patch("src.ml.autonomous_pipeline.InstrumentedTrainer")
@patch("src.ml.autonomous_pipeline.calculate_ks_test")
def test_pipeline_run(mock_ks, mock_trainer_class, mock_get_db, mock_scraper_class, mock_config):
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
    mock_trainer.optimize.return_value = MagicMock(best_params={"max_depth": 3})
    
    # Mock KS
    mock_ks.return_value = (0.1, 0.9)
    
    pipeline = AutonomousMLPipeline(mock_config)
    pipeline.run()
    
    assert mock_scraper.fetch_historical_data.called
    assert mock_session.add.called
    assert mock_trainer.optimize.called
    assert mock_ks.called
