import pytest
import pandas as pd
from src.ml.forecasting.tft_model import PriceTFTModel

def test_tft_prediction_stub():
    model = PriceTFTModel()
    # Should return None as it is not implemented
    assert model.predict(pd.DataFrame()) is None
