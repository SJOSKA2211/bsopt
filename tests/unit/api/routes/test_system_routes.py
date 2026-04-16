from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.index import app
from api.middleware.jwt_validator import require_tier

client = TestClient(app)

@pytest.fixture
def mock_admin_tier():
    # Bypass require_tier dependency
    app.dependency_overrides[require_tier] = lambda: None
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def mock_shm_probe():
    with patch("api.routes.system.SharedMemoryRingBuffer") as mock_shm:
        import struct
        mock_instance = MagicMock()
        # Mock struct.unpack("q", ...) head lookup
        mock_instance.buf = struct.pack("q", 1234) + (b"\x00" * 1024)
        mock_shm.return_value = mock_instance
        yield mock_instance

def test_deep_health_all_up(mock_admin_tier, mock_shm_probe):
    with patch("src.shared.config.settings"), \
         patch("src.shared.utils.cache.get_redis") as mock_redis_getter, \
         patch("aio_pika.connect_robust", new_callable=AsyncMock):
        
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis_getter.return_value = mock_redis
        
        response = client.get("/api/v1/system/health/deep")
        
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["status"] == "operational"
        assert data["probes"]["shm_mesh"]["status"] == "connected"
        assert data["probes"]["redis"]["status"] == "connected"
        assert data["probes"]["rabbitmq"]["status"] == "connected"

def test_deep_health_degraded(mock_admin_tier, mock_shm_probe):
    with patch("src.shared.config.settings"), \
         patch("src.shared.utils.cache.get_redis") as mock_redis_getter, \
         patch("aio_pika.connect_robust", side_effect=Exception("RabbitMQ Down")):
        
        mock_redis_getter.return_value = None
        
        response = client.get("/api/v1/system/health/deep")
        
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["status"] == "degraded"
        assert data["probes"]["rabbitmq"]["status"] == "error"

def test_system_status(mock_admin_tier):
    from src.shared.utils.circuit_breaker import pricing_circuit
    pricing_circuit.failure_count = 5
    
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    assert response.json()["data"]["circuits"]["pricing"]["failure_count"] == 5

@pytest.mark.asyncio
async def test_diagnostics_endpoints(mock_admin_tier):
    mock_db = AsyncMock()
    from src.database import get_async_db
    app.dependency_overrides[get_async_db] = lambda: mock_db
    
    with patch("src.database.crud.get_system_health_dashboard", new_callable=AsyncMock) as mock_health, \
         patch("src.database.crud.get_io_performance_audit", new_callable=AsyncMock) as mock_io:
        
        mock_health.return_value = {"cpu": 10.5}
        mock_io.return_value = {"io_latency": "0.1ms"}
        
        resp_db = client.get("/api/v1/system/diagnostics/db")
        resp_io = client.get("/api/v1/system/diagnostics/io")
        
        assert resp_db.status_code == 200
        assert resp_io.status_code == 200
        assert resp_db.json()["data"]["cpu"] == 10.5
        assert resp_io.json()["data"]["io_latency"] == "0.1ms"
    
    app.dependency_overrides.clear()
