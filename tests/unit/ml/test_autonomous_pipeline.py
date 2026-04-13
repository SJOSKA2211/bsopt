from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.autonomous_pipeline import AutonomousMLPipeline
from src.ml.pipeline import PipelineConfig
from src.ml.training.base import TrainingConfig


@pytest.fixture
def mock_training_config():
    return TrainingConfig(
        framework="xgboost",
        n_estimators=100,
        metadata={"ticker": "AAPL", "study_name": "test_study"}
    )


@pytest.fixture
def sample_df():
    data = {
        "symbol": ["AAPL"] * 10,
        "time": pd.date_range(start="2025-01-01", periods=10, freq="h"),
        "last": np.random.uniform(150, 160, 10),
        "strike": np.random.uniform(140, 170, 10),
        "expiry": pd.date_range(start="2025-02-01", periods=10, freq="h"),
        "implied_volatility": np.random.uniform(0.1, 0.3, 10),
        "option_type": ["call"] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def mock_ml_deps():
    with (
        patch("src.ml.indicators._numba_rsi", return_value=np.zeros(100)),
        patch("src.ml.indicators._numba_macd", return_value=(np.zeros(100), np.zeros(100), np.zeros(100))),
        patch("src.ml.indicators._numba_bbands", return_value=(np.zeros(100), np.zeros(100), np.zeros(100))),
        patch("src.ml.indicators._numba_atr", return_value=np.zeros(100)),
        patch("src.ml.indicators.get_adx", return_value=np.zeros(100)),
        patch("src.ml.pipeline.mlflow"),
        patch("src.ml.trainer.mlflow"),
        patch("src.ml.registry.promote.promote_model"),
    ):
        yield


@pytest.mark.asyncio
async def test_pipeline_init(mock_training_config):
    pipeline = AutonomousMLPipeline(mock_training_config)
    assert pipeline.pipeline_config.symbols == ["AAPL"]
    assert pipeline.training_config.framework == "xgboost"


@pytest.mark.asyncio
async def test_pipeline_run_success(mock_training_config, sample_df):
    pipeline = AutonomousMLPipeline(mock_training_config)
    
    # Mock data pipeline loading
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    features = ["strike", "delta", "gamma", "vega", "iv"]
    meta = {"count": 10}
    
    pipeline.data_pipeline.run = AsyncMock()
    pipeline.data_pipeline.load_latest_data = AsyncMock(return_value=(X, y, features, meta))

    # Mock trainer
    from src.ml.training.base import TrainingResult
    mock_result = TrainingResult(score=0.85, model_path="mock_path")
    pipeline.trainer.train_and_evaluate = MagicMock(return_value=mock_result)
    pipeline.trainer.model = MagicMock() # Set model to trigger promotion logic

    result = await pipeline.run()

    assert result.score == 0.85
    pipeline.data_pipeline.run.assert_called_once()
    pipeline.data_pipeline.load_latest_data.assert_called_once()
    pipeline.trainer.train_and_evaluate.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_shutdown(mock_training_config):
    pipeline = AutonomousMLPipeline(mock_training_config)
    pipeline.data_pipeline.run_shutdown = AsyncMock()
    
    await pipeline.shutdown()
    pipeline.data_pipeline.run_shutdown.assert_called_once()