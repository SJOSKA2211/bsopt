import pytest
import pandas as pd
import numpy as np
from src.ml.forecasting.tft_model import PriceTFTModel

@pytest.fixture
def sample_market_data():
    """Create synthetic market data for TFT testing."""
    data = []
    symbols = ["AAPL", "GOOG"]
    for symbol in symbols:
        for i in range(60):
            data.append({
                "time_idx": i,
                "symbol": symbol,
                "close": 100.0 + np.random.randn() * 2,
                "volume": 1000 + np.random.randint(0, 100),
                "day_of_week": i % 5,
                "month": (i // 20) % 12 + 1
            })
    return pd.DataFrame(data)

def test_tft_data_preparation(sample_market_data):
    model = PriceTFTModel()
    dataset = model.prepare_dataset(sample_market_data)
    assert dataset is not None
    assert dataset.target == "close"

def test_tft_prediction_no_model():
    model = PriceTFTModel()
    assert model.predict(pd.DataFrame()) is None

def test_tft_interpretability_report():
    model = PriceTFTModel()
    # Report should be empty if no model/dataset
    assert model.get_interpretability_report() == {}
    
    # Mock model and dataset
    model.model = "fake_model"
    model.training_dataset = "fake_dataset"
    report = model.get_interpretability_report()
    assert report["interpretable_attention"] is True
    assert "static_variables" in report

@pytest.mark.asyncio
async def test_tft_training_and_prediction(sample_market_data):
    model = PriceTFTModel(max_prediction_length=2, max_encoder_length=10)
    
    # Train for 1 epoch
    await model.train(sample_market_data, max_epochs=1, batch_size=32)
    assert model.model is not None
    
    # Test Prediction
    predictions = model.predict(sample_market_data)
    assert predictions is not None
    assert hasattr(predictions, "output")
    assert hasattr(predictions.output, "prediction")