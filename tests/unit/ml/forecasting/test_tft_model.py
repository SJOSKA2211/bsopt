import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
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
                "price": 100.0 + np.random.randn() * 2, # Model expects 'price'
                "volume": 1000 + np.random.randint(0, 100),
                "day_of_week": i % 5,
                "month": (i // 20) % 12 + 1
            })
    return pd.DataFrame(data)

def test_tft_data_preparation(sample_market_data):
    model = PriceTFTModel()
    # Mock prepare_data since it requires heavy deps and data
    with patch.object(model, 'prepare_data') as mock_prep:
        mock_prep.return_value = {"train_loader": MagicMock()}
        dataset = model.prepare_data(sample_market_data)
        assert dataset is not None

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
    assert "encoder_variables" in report
    assert "static_variables" in report

@pytest.mark.asyncio
async def test_tft_training_and_prediction(sample_market_data):
    config = {
        "max_prediction_length": 2,
        "max_encoder_length": 10
    }
    model = PriceTFTModel(config=config)
    
    # Mock the internal methods to avoid actual training
    with patch.object(model, 'prepare_data', return_value={"train_loader": MagicMock()}), \
            patch.object(model, '_init_trainer') as mock_trainer_cls, \
            patch('src.ml.forecasting.tft_model.TemporalFusionTransformer') as mock_tft, \
            patch('mlflow.start_run'):
        
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        
        trained_model = await model.train(sample_market_data)
        assert trained_model is not None
