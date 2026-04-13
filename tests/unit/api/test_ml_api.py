from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from api.index import app
from api.schemas.ml import InferenceResponse, InferenceRequest
from src.ml.service import get_ml_service, MLService

client = TestClient(app, raise_server_exceptions=False)

@pytest.fixture(autouse=True)
def override_auth():
    from api.middleware.jwt_validator import require_auth
    mock_claims = MagicMock()
    mock_claims.tier = "pro"
    mock_claims.user_id = "test-user-id"
    app.dependency_overrides[require_auth] = lambda: mock_claims
    
    from src.auth.auth import get_current_active_user
    mock_user = MagicMock()
    mock_user.id = "test-user-id"
    mock_user.tier = "pro"
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def mock_ml_service():
    mock = MagicMock(spec=MLService)
    mock.predict = AsyncMock()
    app.dependency_overrides[get_ml_service] = lambda: mock
    return mock

@pytest.fixture
def valid_ml_payload():
    return {
        "underlying_price": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "is_call": 1,
        "moneyness": 1.0,
        "log_moneyness": 0.0,
        "sqrt_time_to_expiry": 1.0,
        "days_to_expiry": 365.0,
        "implied_volatility": 0.2,
    }

def test_proxy_predict_success(mock_ml_service, valid_ml_payload):
    mock_ml_service.predict.return_value = InferenceResponse(
        price=12.5, model_type="grpc_xgb", latency_ms=10.0
    )

    response = client.post("/api/v1/ml/predict", json=valid_ml_payload)
    assert response.status_code == 200
    assert response.json()["data"]["price"] == 12.5
    assert response.json()["data"]["model_type"] == "grpc_xgb"

def test_proxy_predict_service_unavailable(mock_ml_service, valid_ml_payload):
    # In the real code, if gRPC fails, it falls back to Black-Scholes.
    # To test a 503, we need to mock the service to raise an exception that the route doesn't catch
    # OR mock the service to return something that triggers a 503.
    # However, the current route doesn't catch exceptions from ml_service.predict.
    # It depends on global exception handler.
    
    from api.exceptions import ServiceUnavailableException
    mock_ml_service.predict.side_effect = ServiceUnavailableException("ML service unreachable")
    
    response = client.post("/api/v1/ml/predict", json=valid_ml_payload)
    assert response.status_code == 503
    assert "unreachable" in response.json()["message"]

def test_proxy_predict_unexpected_error(mock_ml_service, valid_ml_payload):
    mock_ml_service.predict.side_effect = Exception("Unhandled error")
    
    response = client.post("/api/v1/ml/predict", json=valid_ml_payload)
    assert response.status_code == 500
    assert "Internal server error" in response.json()["message"]

def test_get_predictions_success(mock_ml_service):
    mock_ml_service.predict.return_value = InferenceResponse(
        price=15.0, model_type="grpc_xgb", latency_ms=5.0
    )
    
    response = client.get("/api/v1/ml/predictions?symbol=AAPL")
    assert response.status_code == 200
    assert response.json()["data"]["price"] == 15.0

def test_get_drift_metrics_success(mock_db_session):
    # Drift metrics uses get_model_drift_metrics CRUD
    with patch("api.routes.ml.get_model_drift_metrics") as mock_get_metrics:
        from datetime import datetime, UTC
        from uuid import uuid4
        from api.schemas.ml import DriftMetrics
        
        mock_get_metrics.return_value = [
            DriftMetrics(
                model_id=str(uuid4()),
                window_hour=datetime.now(UTC),
                mae=0.05,
                rmse=0.08,
                prediction_count=100
            )
        ]
        
        # We need to bypass the tier check or mock it
        response = client.get("/api/v1/ml/drift-metrics")
        # Default mock auth in conftest/this file might not have enterprise tier
        # Let's see what happens
        if response.status_code == 403:
            # Update override_auth to have enterprise tier
            from api.middleware.jwt_validator import require_auth
            mock_claims = MagicMock()
            mock_claims.tier = "enterprise"
            app.dependency_overrides[require_auth] = lambda: mock_claims
            response = client.get("/api/v1/ml/drift-metrics")
            
        assert response.status_code == 200
        assert "metrics" in response.json()["data"]