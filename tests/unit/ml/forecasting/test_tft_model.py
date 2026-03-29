from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ml.forecasting.tft_model import PriceTFTModel


@pytest.fixture
def sample_market_data():
    """Create synthetic market data for TFT testing."""
    data = []
    symbols = ["AAPL", "GOOG"]
    for symbol in symbols:
        for i in range(60):
            data.append(
                {
                    "time_idx": i,
                    "symbol": symbol,
                    "close": 100.0 + np.random.randn() * 2,
                    "volume": 1000 + np.random.randint(0, 100),
                    "day_of_week": i % 5,
                    "month": (i // 20) % 12 + 1,
                }
            )
    return pd.DataFrame(data)

def test_tft_data_preparation(sample_market_data):
    model = PriceTFTModel()
    dataset = model.prepare_data(sample_market_data)
    assert dataset is not None
    assert "train_loader" in dataset

def test_tft_prediction_no_model():
    model = PriceTFTModel()
    assert model.predict(pd.DataFrame()) is None

def test_tft_interpretability_report():
    model = PriceTFTModel()
    # Report should be empty if no model/dataset
    assert model.get_interpretability_report() == {}

    # Mock model and dataset
    model.model = MagicMock()
    model.training_dataset = MagicMock()

    # Mock interpret_output
    model.model.interpret_output.return_value = {
        "encoder_variables": [0.1],
        "decoder_variables": [0.2],
        "static_variables": [0.3],
    }
    model.model.predict.return_value = [MagicMock()]

    report = model.get_interpretability_report()
    assert "encoder_variables" in report

@pytest.mark.asyncio
async def test_tft_training_and_prediction(sample_market_data):
    model = PriceTFTModel(config={"max_prediction_length": 2, "max_encoder_length": 10})

    # Train using async path
    with patch("lightning.pytorch.Trainer.fit") as mock_fit:
        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                await model.train_async(sample_market_data, max_epochs=1)
                assert model.config["max_epochs"] == 1
                mock_fit.assert_called()
