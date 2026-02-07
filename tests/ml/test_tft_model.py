from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.forecasting.tft_model import TFTModel
from src.ml.utils.validation import WalkForwardValidator


@pytest.fixture
def sample_data():
    """🚀 SINGULARITY: Structured synthetic data for TFT validation."""
    dates = pd.date_range(start="2023-01-01", periods=500, freq="h")
    symbols = ["AAPL", "GOOGL"]
    data = []
    for symbol in symbols:
        # 🚀 SOTA: Sine wave + trend to test structural learning
        base_price = 100.0
        for i, date in enumerate(dates):
            price = (
                base_price + 5 * np.sin(i / 24.0) + 0.01 * i + np.random.randn() * 0.1
            )
            data.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "price": price,
                    "volume": 1000 + np.random.randint(-100, 100),
                    "day_of_week": date.dayofweek,
                    "hour": date.hour,
                }
            )
    return pd.DataFrame(data)


def test_tft_temporal_validation(sample_data, tft_config):
    """🚀 SINGULARITY: Test TFT with WalkForwardValidator."""
    model = TFTModel(config=tft_config)
    validator = WalkForwardValidator(n_splits=3)

    X = sample_data.values
    for train_idx, test_idx in validator.split(X):
        train_df = sample_data.iloc[train_idx]
        test_df = sample_data.iloc[test_idx]

        # Verify no leakage
        assert train_df["date"].max() < test_df["date"].min()

        # Dry run data prep
        processed = model.prepare_data(train_df)
        assert "train_loader" in processed


@pytest.fixture
def tft_config():
    """TFT model configuration."""
    return {
        "input_chunk_length": 24,
        "output_chunk_length": 12,
        "hidden_size": 8,
        "lstm_layers": 1,
        "num_attention_heads": 2,
        "dropout": 0.1,
        "batch_size": 16,
        "max_epochs": 1,
    }


def test_tft_model_initialization(tft_config):
    """Test that TFTModel initializes with correct parameters."""
    model = TFTModel(config=tft_config)
    assert model.config == tft_config
    assert model.model is None


def test_prepare_data_logic(sample_data, tft_config):
    """Test the data preparation logic for TFT."""
    model = TFTModel(config=tft_config)
    processed_data = model.prepare_data(sample_data)

    assert "train_loader" in processed_data
    assert "val_loader" in processed_data
    assert processed_data["group_ids"] == ["symbol"]
    assert model.training_dataset is not None


@pytest.mark.asyncio
@patch("src.ml.forecasting.tft_model.mlflow")
async def test_tft_model_train(mock_mlflow, sample_data, tft_config):
    """Test the training process and MLflow integration."""
    model = TFTModel(config=tft_config)

    # Mock the internal trainer to avoid actual heavy training
    with patch.object(model, "_init_trainer") as mock_init_trainer:
        mock_trainer = MagicMock()
        mock_init_trainer.return_value = mock_trainer

        await model.train(sample_data)

        mock_mlflow.start_run.assert_called_once()
        mock_trainer.fit.assert_called_once()
        mock_mlflow.log_params.assert_called()


def test_tft_model_predict(sample_data, tft_config):
    """Test the prediction functionality."""
    model = TFTModel(config=tft_config)
    model.model = MagicMock()

    # Mock prediction output
    mock_prediction = np.random.randn(12, 1)  # 12 hours ahead
    model.model.predict.return_value = mock_prediction

    # Simple mock for data to avoid TimeSeriesDataSet complexity in this unit test
    predictions = model.predict(MagicMock())

    assert predictions is not None
    model.model.predict.assert_called_once()


def test_get_interpretability_report_trained(sample_data, tft_config):
    """Test that TFT provides feature importance/attention insights when trained."""
    model = TFTModel(config=tft_config)
    model.model = MagicMock()
    model.training_dataset = MagicMock()

    # Mock interpretation output
    model.model.predict.return_value = (None, MagicMock())
    model.model.interpret_output.return_value = MagicMock()

    importance = model.get_interpretability_report()

    assert "price" in importance["encoder_variables"]
    assert importance["encoder_variables"]["price"] == 0.6


def test_tft_model_invalid_data(tft_config):
    """Test data preparation with missing columns."""
    model = TFTModel(config=tft_config)
    bad_data = pd.DataFrame({"wrong": [1, 2, 3]})
    with pytest.raises(KeyError):
        model.prepare_data(bad_data)


def test_tft_model_predict_not_trained(tft_config):
    """Test predict when model is not trained."""
    model = TFTModel(config=tft_config)
    assert model.predict(pd.DataFrame()) is None


def test_get_interpretability_report_not_trained(tft_config):
    """Test interpretability report when model is not trained."""
    model = TFTModel(config=tft_config)
    assert model.get_interpretability_report() == {}


def test_prepare_data_with_volume(sample_data, tft_config):
    """Test data preparation specifically including volume."""
    model = TFTModel(config=tft_config)
    # Ensure volume is present
    assert "volume" in sample_data.columns
    processed_data = model.prepare_data(sample_data)
    assert "train_loader" in processed_data
