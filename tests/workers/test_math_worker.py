import pytest
from unittest.mock import MagicMock, patch, AsyncMock
# We import recalibrate_symbol which is a Task object (Proxy)
from src.workers.math_worker import health_check, recalibrate_symbol
import json
import time
import uuid

@pytest.fixture
def mock_worker_deps():
    with patch("src.workers.math_worker.redis_client") as mock_redis, \
         patch("src.workers.math_worker.MarketDataRouter") as mock_router, \
         patch("src.workers.math_worker.HestonCalibrator") as mock_calib, \
         patch("src.workers.math_worker.get_db_context") as mock_db, \
         patch("src.workers.math_worker.push_metrics") as mock_push, \
         patch("src.shared.observability.HESTON_PARAMS_FRESHNESS", MagicMock()), \
         patch("src.shared.observability.HESTON_FELLER_MARGIN", MagicMock()), \
         patch("src.shared.observability.CALIBRATION_DURATION", MagicMock()), \
         patch("src.shared.observability.MODEL_RMSE", MagicMock()), \
         patch("src.shared.observability.HESTON_R_SQUARED", MagicMock()):
        
        yield mock_redis, mock_router, mock_calib, mock_db, mock_push

def test_health_check(mock_worker_deps):
    mock_redis, _, _, _, _ = mock_worker_deps
    data = {"timestamp": time.time() - 100}
    mock_redis.get.return_value = json.dumps(data)
    status = health_check()
    assert status["SPY"] == "healthy"

def test_recalibrate_symbol_success(mock_worker_deps):
    mock_redis, mock_router, mock_calib, mock_db, mock_push = mock_worker_deps
    
    mock_router.return_value.get_option_chain_snapshot = AsyncMock(return_value=[
        {"T": 0.1, "strike": 400.0, "spot": 400.0, "price": 10.0, "bid": 9.9, "ask": 10.1, "volume": 100, "open_interest": 1000, "option_type": "call"}
    ])
    
    mock_calib_instance = mock_calib.return_value
    mock_params = MagicMock()
    mock_params.v0 = 0.1; mock_params.kappa = 0.1; mock_params.theta = 0.1; mock_params.sigma = 0.1; mock_params.rho = 0.1
    mock_calib_instance.calibrate.return_value = (mock_params, {"rmse": 0.001, "r_squared": 0.99, "num_options": 1})
    mock_calib_instance.calibrate_surface.return_value = {0.1: [1, 2, 3, 4, 5]}
    
    # recalibrate_symbol is a Task. Calling it as a function recalibrate_symbol("SPY") 
    # should work if it's bound and Celery handles it.
    
    # We need to mock the 'request' attribute of the Task instance.
    # Celery Task.request is a proxy to current_task.request.
    with patch("celery.app.task.Task.request") as mock_req:
        mock_req.id = "test-id"
        result = recalibrate_symbol("SPY")
    
    assert result["status"] == "success"
    mock_redis.setex.assert_called()

def test_recalibrate_symbol_no_data(mock_worker_deps):
    mock_redis, mock_router, _, _, _ = mock_worker_deps
    mock_router.return_value.get_option_chain_snapshot = AsyncMock(return_value=None)
    
    with patch("celery.app.task.Task.request") as mock_req:
        mock_req.id = "test-id"
        result = recalibrate_symbol("SPY")
    assert result["status"] == "failed"

def test_recalibrate_symbol_exception(mock_worker_deps):
    mock_redis, mock_router, _, _, _ = mock_worker_deps
    mock_router.return_value.get_option_chain_snapshot.side_effect = Exception("Boom")
    
    # Mock retry on the task instance
    with patch("celery.app.task.Task.request") as mock_req, \
         patch.object(recalibrate_symbol, "retry", side_effect=Exception("Retry") ):
        mock_req.id = "test-id"
        mock_req.retries = 0
        with pytest.raises(Exception, match="Retry"):
            recalibrate_symbol("SPY")