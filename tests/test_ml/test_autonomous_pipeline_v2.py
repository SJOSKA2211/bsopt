from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.autonomous_pipeline import AutonomousMLPipeline


@pytest.fixture
def pipeline_config():
    return {
        "api_key": "mock",
        "db_url": "sqlite:///:memory:",
        "ticker": "AAPL",
        "study_name": "test_study",
        "n_trials": 1,
        "framework": "xgboost",
    }


def test_pipeline_init(pipeline_config):
    with (
        patch("src.ml.autonomous_pipeline.create_engine"),
        patch("src.ml.autonomous_pipeline.Base.metadata.create_all"),
    ):
        pipeline = AutonomousMLPipeline(pipeline_config)
        assert pipeline.ticker == "AAPL"


def test_generate_features(pipeline_config):
    with (
        patch("src.ml.autonomous_pipeline.create_engine"),
        patch("src.ml.autonomous_pipeline.Base.metadata.create_all"),
    ):
        pipeline = AutonomousMLPipeline(pipeline_config)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2023-01-01", periods=100, freq="1min"
                ),
                "close": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(110, 115, 100),
                "low": np.random.uniform(95, 100, 100),
                "volume": np.random.uniform(1000, 2000, 100),
            }
        )

        featured_df = pipeline.generate_features(df)
        assert "log_return" in featured_df.columns
        assert "volatility" in featured_df.columns


@pytest.mark.asyncio
async def test_get_current_model_performance(pipeline_config):
    with (
        patch("src.ml.autonomous_pipeline.create_engine"),
        patch("src.ml.autonomous_pipeline.Base.metadata.create_all"),
    ):
        pipeline = AutonomousMLPipeline(pipeline_config)

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0.85
        mock_session.execute.return_value = mock_result

        perf = await pipeline.get_current_model_performance(mock_session)
        assert perf == 0.85
