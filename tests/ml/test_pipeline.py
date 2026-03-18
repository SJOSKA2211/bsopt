from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.pipeline import MLPipeline


@pytest.fixture
def mock_config():
    return {
        "api_key": "test_key",
        "provider": "mock",
        "db_url": "sqlite:///:memory:",
        "ticker": "AAPL",
        "study_name": "test_study",
        "n_trials": 1,
        "framework": "xgboost",
    }


@pytest.fixture
def mock_df():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates.astype(int) // 10**9,
            "open": np.random.rand(100) * 100,
            "high": np.random.rand(100) * 110,
            "low": np.random.rand(100) * 90,
            "close": np.random.rand(100) * 100,
            "volume": np.random.randint(1000, 10000, 100),
        }
    )


@pytest.mark.asyncio
async def test_pipeline_run(mock_config, mock_df):
    with patch("src.ml.pipeline.create_engine"):
        with patch("src.ml.pipeline.Base.metadata.create_all"):
            with patch("src.ml.pipeline.MarketDataScraper") as MockScraper:
                with patch("src.ml.pipeline.get_async_db_context") as mock_db_ctx:
                    with patch("src.ml.pipeline.DriftTrigger") as MockDrift:
                        with patch("src.ml.pipeline.InstrumentedTrainer") as MockTrainer:
                            with patch(
                                "src.database.crud.bulk_insert_market_ticks",
                                new_callable=AsyncMock,
                            ) as mock_bulk_insert:
                                with patch(
                                    "src.workers.tasks.ml_tasks.optimize_model_task.delay"
                                ) as mock_task:
                                    # Setup Scraper
                                    mock_scraper_instance = MockScraper.return_value
                                    mock_scraper_instance.fetch_historical_data = AsyncMock(
                                        return_value=mock_df
                                    )

                                    # Setup DB
                                    mock_session = AsyncMock()
                                    mock_db_ctx.return_value.__aenter__.return_value = mock_session

                                    # Setup Drift
                                    mock_drift_instance = MockDrift.return_value
                                    mock_drift_instance.should_retrain.return_value = (
                                        True,
                                        "drift detected",
                                    )

                                    # Setup Trainer
                                    mock_trainer_instance = MockTrainer.return_value
                                    mock_study = MagicMock()
                                    mock_study.best_value = 0.9
                                    mock_study.best_params = {"n_estimators": 100}
                                    mock_trainer_instance.optimize.return_value = mock_study

                                    pipeline = MLPipeline(mock_config)

                                    # Run
                                    study = await pipeline.run()

                                    assert study == mock_study
                                    mock_bulk_insert.assert_called_once()
                                    mock_trainer_instance.optimize.assert_called_once()
                                    mock_task.assert_called()


def test_feature_generation(mock_config, mock_df):
    with patch("src.ml.pipeline.create_engine"):
        pipeline = MLPipeline(mock_config)
        df_featured = pipeline.generate_features(mock_df)

        assert "RSI_14" in df_featured.columns
        assert "MACD_12_26_9" in df_featured.columns
        assert "BBU_20_2.0" in df_featured.columns
        assert "ATR_14" in df_featured.columns
        assert "ADX_14" in df_featured.columns
        assert not df_featured.isnull().values.any()


@pytest.mark.asyncio
async def test_get_current_model_performance(mock_config):
    with patch("src.ml.pipeline.create_engine"):
        pipeline = MLPipeline(mock_config)
        mock_session = AsyncMock()

        # Create a sync Mock for the result object
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0.85

        # AsyncSession.execute is async, so it returns the result when awaited
        mock_session.execute.return_value = mock_result

        perf = await pipeline.get_current_model_performance(mock_session)
        assert perf == 0.85
