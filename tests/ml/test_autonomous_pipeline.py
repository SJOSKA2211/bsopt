import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.ml.autonomous_pipeline import AutonomousMLPipeline

@pytest.fixture
def mock_config():
    return {
        "api_key": "test_key",
        "provider": "mock",
        "db_url": "sqlite:///:memory:",
        "ticker": "AAPL",
        "study_name": "test_study",
        "n_trials": 1,
        "framework": "xgboost"
    }

@pytest.fixture
def mock_scraper():
    with patch("src.ml.autonomous_pipeline.MarketDataScraper") as MockScraper:
        scraper = MagicMock()
        MockScraper.return_value = scraper
        
        # Create a dummy DataFrame
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "timestamp": dates.astype(np.int64) // 10**9,
            "open": np.random.rand(10) * 100,
            "high": np.random.rand(10) * 100,
            "low": np.random.rand(10) * 100,
            "close": np.random.rand(10) * 100,
            "volume": np.random.randint(100, 1000, 10)
        })
        scraper.fetch_historical_data.return_value = df
        yield scraper

@pytest.fixture
def mock_trainer():
    with patch("src.ml.autonomous_pipeline.InstrumentedTrainer") as MockTrainer:
        trainer = MagicMock()
        MockTrainer.return_value = trainer
        
        study = MagicMock()
        study.best_value = 0.85
        trainer.optimize.return_value = study
        trainer.train_and_evaluate.return_value = 0.85
        yield trainer

@pytest.fixture
def mock_observability():
    with patch("src.ml.autonomous_pipeline.push_metrics"), \
         patch("src.ml.autonomous_pipeline.DATA_DRIFT_SCORE"), \
         patch("src.ml.autonomous_pipeline.KS_TEST_SCORE"), \
         patch("src.ml.autonomous_pipeline.setup_logging"):
        yield

@pytest.fixture
def mock_db():
    with patch("src.ml.autonomous_pipeline.create_engine") as mock_create_engine, \
         patch("src.ml.autonomous_pipeline.get_db_session") as mock_get_session, \
         patch("src.ml.autonomous_pipeline.Base.metadata.create_all"):
        
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        
        yield mock_session

def test_pipeline_run(mock_config, mock_scraper, mock_trainer, mock_observability, mock_db):
    pipeline = AutonomousMLPipeline(mock_config)
    study = pipeline.run()
    
    # Verify interactions
    mock_scraper.fetch_historical_data.assert_called()
    mock_db.add.assert_called() # Should persist data
    mock_db.commit.assert_called()
    mock_trainer.optimize.assert_called()
    assert study.best_value == 0.85

def test_pipeline_scrape_retry(mock_config, mock_scraper, mock_trainer, mock_observability, mock_db):
    # Simulate scrape failure on first try
    mock_scraper.fetch_historical_data.side_effect = [Exception("API Error"), mock_scraper.fetch_historical_data.return_value]
    
    pipeline = AutonomousMLPipeline(mock_config)
    
    # We need to mock the fallback scraper creation inside run()
    # The code does: self.scraper = MarketDataScraper(...)
    # Since we patched MarketDataScraper class, calling it again returns a new mock.
    # We need to ensure the second mock returns data.
    
    # Reset side_effect for the *class* return value?
    # No, run() calls `self.scraper.fetch...`.
    # Then creates NEW scraper instance.
    
    # Let's just allow the exception to happen and verify logging?
    # The code catches exception and retries with mock provider.
    
    # To test this, we need the second instance of MarketDataScraper to work.
    # Our fixture sets return_value for the CLASS. So any instance created returns the same mock?
    # No, `MockScraper.return_value = scraper` means `MarketDataScraper()` returns `scraper`.
    # So both self.scraper and the fallback scraper will be the SAME object `scraper`.
    # So `scraper.fetch_historical_data` will be called twice.
    # First time raises, second time returns DF.
    
    # We set side_effect on the method instance.
    mock_scraper.fetch_historical_data.side_effect = [Exception("Fail"), mock_scraper.fetch_historical_data.return_value]
    
    pipeline.run()
    assert mock_scraper.fetch_historical_data.call_count == 2

def test_pipeline_failure(mock_config, mock_scraper, mock_trainer, mock_observability, mock_db):
    mock_scraper.fetch_historical_data.side_effect = Exception("Fatal Error")
    
    # Ensure it raises
    with pytest.raises(Exception):
        pipeline = AutonomousMLPipeline(mock_config)
        # We need to make sure the retry logic also fails if we want to test total failure
        # Or mock the retry scraper to fail too.
        # But here, we can just let the retry fail (since we reuse the mock which raises exception)
        # Wait, if we reuse the mock, and side_effect is just one exception, next call might work if iterator exhausted?
        # If side_effect is an Exception class, it raises every time.
        
        pipeline.run()

def test_objective_function(mock_config, mock_scraper, mock_trainer, mock_observability, mock_db):
    # To test the objective function defined inside run(), we need to execute run()
    # and ensure optimize calls it.
    # We can mock optimize to call the function passed to it.
    
    def side_effect_optimize(func, n_trials):
        trial = MagicMock()
        trial.suggest_int.return_value = 10
        trial.suggest_float.return_value = 0.1
        func(trial) # Execute objective
        return MagicMock(best_value=0.9)
        
    mock_trainer.optimize.side_effect = side_effect_optimize
    
    pipeline = AutonomousMLPipeline(mock_config)
    pipeline.run()
    
    mock_trainer.train_and_evaluate.assert_called()

def test_pytorch_framework(mock_config, mock_scraper, mock_trainer, mock_observability, mock_db):
    mock_config["framework"] = "pytorch"
    pipeline = AutonomousMLPipeline(mock_config)
    
    def side_effect_optimize(func, n_trials):
        trial = MagicMock()
        trial.suggest_int.return_value = 10
        trial.suggest_float.return_value = 0.001
        func(trial)
        return MagicMock(best_value=0.9)
        
    mock_trainer.optimize.side_effect = side_effect_optimize
    pipeline.run()
    
    call_args = mock_trainer.train_and_evaluate.call_args
    assert call_args[0][2]["framework"] == "pytorch"
