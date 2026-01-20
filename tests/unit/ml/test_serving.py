import pytest
import sys
from unittest.mock import MagicMock, patch

# Mock onnxruntime if not available
try:
    import onnxruntime
except ImportError:
    sys.modules["onnxruntime"] = MagicMock()

from src.ml.serving.serve import predict, state
from src.api.schemas.ml import InferenceRequest

@pytest.mark.asyncio
async def test_predict_xgb_success():
    # Mock XGB model
    mock_model = MagicMock()
    mock_model.predict.return_value = [15.5]
    
    with patch.dict(state, {"xgb_model": mock_model}):
        request = InferenceRequest(
            underlying_price=100.0,
            strike=105.0,
            time_to_expiry=0.5,
            is_call=1,
            moneyness=0.95,
            log_moneyness=-0.05,
            sqrt_time_to_expiry=0.707,
            days_to_expiry=182.5,
            implied_volatility=0.25
        )
        
        response = await predict(request, model_type="xgb")
        assert response.data.price == 15.5
        assert response.data.model_type == "xgb"

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