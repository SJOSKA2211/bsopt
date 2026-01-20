import pytest
import numpy as np
import sys
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mock onnxruntime if not available
try:
    import onnxruntime
except ImportError:
    sys.modules["onnxruntime"] = MagicMock()

from src.ml.serving.serve import app, predict, state, load_xgb_model
from src.api.schemas.ml import InferenceRequest

@pytest.fixture(autouse=True)
def mock_models(mocker):
    mocker.patch("src.ml.serving.serve.mlflow.pyfunc.load_model", side_effect=Exception("Mocked Load Fail"))
    mocker.patch("src.ml.serving.serve.ort.InferenceSession", side_effect=Exception("Mocked Load Fail"))
    yield

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def reset_state():
    # Setup
    state["xgb_model"] = None
    state["nn_ort_session"] = None
    yield
    # Teardown
    state["xgb_model"] = None
    state["nn_ort_session"] = None

def test_predict_xgb_success(client, mocker):
    # Mock model
    mock_model = mocker.Mock()
    mock_model.predict.return_value = np.array([10.5])
    
    # Update state directly
    state["xgb_model"] = mock_model
    print(f"DEBUG: State xgb_model is {state['xgb_model']}")
    
    payload = {
        "underlying_price": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "is_call": 1,
        "moneyness": 1.0,
        "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0,
        "days_to_expiry": 365.0,
        "implied_volatility": 0.2,
        "volatility": 0.2,
        "rate": 0.05
    }
    response = client.post("/predict?model_type=xgb", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["price"] == 10.5

def test_predict_nn_success(client, mocker):
    # Mock ONNX session
    mock_session = mocker.Mock()
    mock_session.get_inputs.return_value = [mocker.Mock(name="input")]
    mock_session.run.return_value = [np.array([[5.2]])]
    
    state["nn_ort_session"] = mock_session

    payload = {
        "underlying_price": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "is_call": 1,
        "moneyness": 1.0,
        "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0,
        "days_to_expiry": 365.0,
        "implied_volatility": 0.2,
        "volatility": 0.2,
        "rate": 0.05
    }
    response = client.post("/predict?model_type=nn", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["price"] == 5.2

def test_predict_xgb_unavailable(client):
    state["xgb_model"] = None
    payload = {
        "underlying_price": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "is_call": 1,
        "moneyness": 1.0,
        "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0,
        "days_to_expiry": 365.0,
        "implied_volatility": 0.2,
        "rate": 0.05
    }
    response = client.post("/predict?model_type=xgb", json=payload)
    assert response.status_code == 503

def test_predict_nn_unavailable(client):
    state["nn_ort_session"] = None
    payload = {
        "underlying_price": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "is_call": 1,
        "moneyness": 1.0,
        "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0,
        "days_to_expiry": 365.0,
        "implied_volatility": 0.2,
        "rate": 0.05
    }
    response = client.post("/predict?model_type=nn", json=payload)
    assert response.status_code == 503

def test_health_check(client):
    state["xgb_model"] = MagicMock()
    state["nn_ort_session"] = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["models"]["xgb"] is True
    assert response.json()["models"]["nn"] is False

@pytest.mark.asyncio
async def test_load_models_failure(mocker):
    # Ensure state is None before test (reset_state should handle this, but for sanity)
    state["xgb_model"] = None
    
    with patch("src.ml.serving.serve.mlflow.pyfunc.load_model", side_effect=Exception("Load fail")):
        await load_xgb_model()
        assert state["xgb_model"] is None

@pytest.mark.asyncio
async def test_predict_unsupported_model():
    request = InferenceRequest(
        underlying_price=100.0, strike=100.0, time_to_expiry=1.0, is_call=1,
        moneyness=1.0, log_moneyness=0.0, sqrt_time_to_expiry=1.0, 
        days_to_expiry=365.0, implied_volatility=0.2
    )
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await predict(request, model_type="invalid")
    assert exc.value.status_code == 400