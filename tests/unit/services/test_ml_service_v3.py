import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.schemas.ml import InferenceRequest
from src.ml_service import MLService


@pytest.fixture
def mock_grpc():
    with patch("src.ml_service.Channel") as mock_chan:
        with patch("src.ml_service.MLInferenceStub") as mock_stub:
            yield mock_chan, mock_stub


@pytest.fixture
def mock_shm():
    with patch("src.ml_service.SHMManager") as mock:
        shm = mock.return_value
        shm.name = "ml_shm"
        yield shm


@pytest.mark.asyncio
async def test_predict_shm_success(mock_grpc, mock_shm):
    mock_chan, mock_stub_cls = mock_grpc
    mock_stub = mock_stub_cls.return_value
    mock_response = MagicMock()
    mock_response.price = 5.0
    mock_response.model_type = "xgb"
    mock_stub.Predict = AsyncMock(return_value=mock_response)

    service = MLService()
    # Fixed with all required fields
    req = InferenceRequest(
        underlying_price=100.0,
        strike=100.0,
        time_to_expiry=0.1,
        is_call=True,
        moneyness=1.0,
        log_moneyness=0.0,
        sqrt_time_to_expiry=math.sqrt(0.1),
        days_to_expiry=36.5,
        implied_volatility=0.2,
    )

    res = await service.predict(req)
    assert res.price == 5.0
    assert mock_shm.write.called


@pytest.mark.asyncio
async def test_predict_shm_failure_fallback(mock_grpc, mock_shm):
    mock_chan, mock_stub_cls = mock_grpc
    mock_stub = mock_stub_cls.return_value
    mock_shm.write.side_effect = Exception("SHM Full")

    mock_response = MagicMock()
    mock_response.price = 4.5
    mock_response.model_type = "fallback"
    mock_stub.Predict = AsyncMock(return_value=mock_response)

    service = MLService()
    req = InferenceRequest(
        underlying_price=100.0,
        strike=100.0,
        time_to_expiry=0.1,
        is_call=True,
        moneyness=1.0,
        log_moneyness=0.0,
        sqrt_time_to_expiry=0.316,
        days_to_expiry=36.5,
        implied_volatility=0.2,
    )

    res = await service.predict(req)
    assert res.price == 4.5
    assert mock_stub.Predict.call_count == 1


@pytest.mark.asyncio
async def test_ml_service_lifecycle(mock_grpc, mock_shm):
    mock_chan_cls, _ = mock_grpc
    mock_chan = mock_chan_cls.return_value
    service = MLService()
    await service.close()
    assert mock_chan.close.called
    assert mock_shm.close.called
