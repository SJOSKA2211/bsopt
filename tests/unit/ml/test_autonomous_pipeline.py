from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.autonomous_pipeline import AutonomousMLPipeline


@pytest.fixture
def mock_config():
    return {
        "api_key": "test_key",
        "db_url": "sqlite:///:memory:",
        "ticker": "AAPL",
        "study_name": "test_study",
        "n_trials": 2,
        "framework": "xgboost",
        "provider": "mock",
    }


@pytest.fixture
def sample_df():
    data = {
        "timestamp": pd.date_range(start="2025-01-01", periods=100, freq="h"),
        "close": np.random.uniform(150, 160, 100),
        "high": np.random.uniform(160, 165, 100),
        "low": np.random.uniform(145, 150, 100),
        "volume": np.random.randint(1000, 5000, 100),
    }
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def mock_indicators():
    with (
        patch("src.ml.autonomous_pipeline._numba_rsi", return_value=np.zeros(100)),
        patch(
            "src.ml.autonomous_pipeline._numba_macd",
            return_value=(np.zeros(100), np.zeros(100), np.zeros(100)),
        ),
        patch(
            "src.ml.autonomous_pipeline._numba_bbands",
            return_value=(np.zeros(100), np.zeros(100), np.zeros(100)),
        ),
        patch("src.ml.autonomous_pipeline._numba_atr", return_value=np.zeros(100)),
        patch("src.ml.autonomous_pipeline._numba_adx", return_value=np.zeros(100)),
        patch("src.ml.autonomous_pipeline.push_metrics"),
        patch("src.ml.trainer.ExperimentTracker"),
        patch("src.ml.tracker.mlflow"),
    ):
        yield


@pytest.mark.asyncio
async def test_pipeline_init(mock_config):
    pipeline = AutonomousMLPipeline(mock_config)
    assert pipeline.ticker == "AAPL"
    assert pipeline.framework == "xgboost"


@pytest.mark.asyncio
async def test_generate_features(mock_config, sample_df):
    pipeline = AutonomousMLPipeline(mock_config)
    df_featured = pipeline.generate_features(sample_df)
    assert "RSI_14" in df_featured.columns
    assert "volatility" in df_featured.columns
    assert not df_featured.isnull().values.any()


@pytest.mark.asyncio
async def test_pipeline_run_success(mock_config, sample_df):
    with patch("src.ml.autonomous_pipeline.DriftTrigger") as mock_trigger_cls:
        mock_trigger = mock_trigger_cls.return_value
        mock_trigger.should_retrain.return_value = (True, "drift detected")

        pipeline = AutonomousMLPipeline(mock_config)
        pipeline._fetch_data = AsyncMock(return_value=sample_df)
        pipeline._persist_data = AsyncMock()

        # Mock study
        mock_study = MagicMock()
        mock_study.best_value = 0.85
        mock_study.best_params = {"n_estimators": 100}

        with patch(
            "src.ml.autonomous_pipeline.InstrumentedTrainer"
        ) as mock_trainer_cls:
            mock_trainer = mock_trainer_cls.return_value
            mock_trainer.optimize = MagicMock(return_value=mock_study)

            # Mock DB performance check
            pipeline.get_current_model_performance = AsyncMock(return_value=0.7)

            # Mock model export task
            with patch("src.tasks.ml_tasks.optimize_model_task.delay") as mock_task:
                result = await pipeline.run()

                assert result == mock_study
                pipeline._fetch_data.assert_called_once()
                pipeline._persist_data.assert_called_once()
                mock_task.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_no_retrain(mock_config, sample_df):
    with patch("src.ml.autonomous_pipeline.DriftTrigger") as mock_trigger_cls:
        mock_trigger = mock_trigger_cls.return_value
        mock_trigger.should_retrain.return_value = (False, "no drift")

        pipeline = AutonomousMLPipeline(mock_config)
        pipeline._fetch_data = AsyncMock(return_value=sample_df)
        pipeline._persist_data = AsyncMock()

        result = await pipeline.run()
        assert result is None


@pytest.mark.asyncio
async def test_get_current_model_performance_error(mock_config):
    pipeline = AutonomousMLPipeline(mock_config)
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(side_effect=Exception("DB fail"))

    perf = await pipeline.get_current_model_performance(mock_session)
    assert perf is None
