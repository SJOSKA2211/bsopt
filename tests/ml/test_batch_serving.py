import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.ml.serving.serve import app, state
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

client = TestClient(app)

@pytest.fixture
def mock_xgb_model():
    model = MagicMock()
    # Mock predict to return a list of floats
    model.predict.return_value = np.array([10.5, 11.0])
    return model

@pytest.fixture
def mock_onnx_session():
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    # Mock run to return a list of lists (standard ONNX output format)
    session.run.return_value = [np.array([[10.5], [11.0]])]
    return session

def test_predict_batch_xgb(mock_xgb_model):
    state["xgb_model"] = mock_xgb_model
    state["nn_ort_session"] = None
    
    payload = {
        "requests": [
            {
                "underlying_price": 100.0,
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "is_call": 1,
                "moneyness": 1.0,
                "log_moneyness": 0.0,
                "sqrt_time_to_expiry": 1.0,
                "days_to_expiry": 365.0,
                "implied_volatility": 0.2
            },
            {
                "underlying_price": 105.0,
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "is_call": 1,
                "moneyness": 1.05,
                "log_moneyness": 0.05,
                "sqrt_time_to_expiry": 1.0,
                "days_to_expiry": 365.0,
                "implied_volatility": 0.2
            }
        ]
    }
    
    response = client.post("/predict/batch?model_type=xgb", json=payload)
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["price"] == 10.5
    assert data["predictions"][1]["price"] == 11.0
    assert data["predictions"][0]["model_type"] == "xgb"

def test_predict_batch_nn(mock_onnx_session):
    state["xgb_model"] = None
    state["nn_ort_session"] = mock_onnx_session
    
    payload = {
        "requests": [
            {
                "underlying_price": 100.0,
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "is_call": 1,
                "moneyness": 1.0,
                "log_moneyness": 0.0,
                "sqrt_time_to_expiry": 1.0,
                "days_to_expiry": 365.0,
                "implied_volatility": 0.2
            },
            {
                "underlying_price": 105.0,
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "is_call": 1,
                "moneyness": 1.05,
                "log_moneyness": 0.05,
                "sqrt_time_to_expiry": 1.0,
                "days_to_expiry": 365.0,
                "implied_volatility": 0.2
            }
        ]
    }
    
    response = client.post("/predict/batch?model_type=nn", json=payload)
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["price"] == 10.5
    assert data["predictions"][1]["price"] == 11.0
    assert data["predictions"][0]["model_type"] == "nn"
