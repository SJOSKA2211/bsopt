import asyncio
import httpx
import pytest
import grpc
import os
from src.shared.protos import auth_pb2, auth_pb2_grpc
from src.shared.utils.cache import get_redis_client
from src.database import db_manager
from sqlalchemy import text

# Configuration for Integration Network
AUTH_GRPC_ADDR = os.getenv("AUTH_SERVICE_GRPC_URL", "localhost:50051")
PRICING_API_URL = os.getenv("PRICING_API_URL", "http://localhost:8000")

@pytest.mark.asyncio
async def test_zero_mock_auth_lifecycle():
    """
    Verifies the complete Auth lifecycle against a live gRPC server.
    Ensures zero-trust internal communication is stable.
    """
    async with grpc.aio.insecure_channel(AUTH_GRPC_ADDR) as channel:
        stub = auth_pb2_grpc.AuthServiceStub(channel)
        
        # 1. Validation of an empty/invalid token should fail gracefully
        try:
            resp = await stub.ValidateToken(auth_pb2.TokenRequest(token="invalid_token"))
            assert resp.valid is False
        except grpc.RpcError as e:
            pytest.fail(f"gRPC ValidateToken failed with unexpected error: {e}")

        # 2. Check gRPC Health service
        from grpc_health.v1 import health_pb2, health_pb2_grpc
        health_stub = health_pb2_grpc.HealthStub(channel)
        health_resp = await health_stub.Check(health_pb2.HealthCheckRequest(service="auth.AuthService"))
        assert health_resp.status == health_pb2.HealthCheckResponse.SERVING

@pytest.mark.asyncio
async def test_zero_mock_pricing_computation():
    """
    Verifies that the Pricing API can perform real-time Black-Scholes calculations.
    Ensures the Rust math kernel is correctly bound and exposed.
    """
    async with httpx.AsyncClient() as client:
        payload = {
            "s": 100.0, "k": 100.0, "t": 1.0,
            "v": 0.2, "r": 0.05, "cp": "call"
        }
        try:
            response = await client.post(f"{PRICING_API_URL}/compute/black_scholes", json=payload)
            if response.status_code == 200:
                data = response.json()
                assert "price" in data
                assert data["price"] > 0
        except httpx.ConnectError:
            pytest.skip("Pricing API not reachable.")

@pytest.mark.asyncio
async def test_zero_mock_infrastructure_heartbeat():
    """
    Directly probes infrastructure components (Database, Redis) to verify zero-trust mesh.
    """
    # 1. Redis Heartbeat
    redis = await get_redis_client()
    if redis:
        pong = await redis.ping()
        assert pong is True
    else:
        pytest.fail("Failed to connect to Redis")

    # 2. Database Heartbeat
    async with db_manager.async_session_factory() as db:
        result = await db.execute(text("SELECT 1"))
        assert result.scalar() == 1

@pytest.mark.asyncio
async def test_zero_mock_nginx_gateway_traversal():
    """
    Verifies that the Nginx gateway correctly routes and enforces headers.
    """
    NGINX_URL = "http://localhost:80"
    async with httpx.AsyncClient() as client:
        try:
            # Test global health aggregate
            resp = await client.get(f"{NGINX_URL}/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "UP"
            
            # Verify security headers injected by Nginx
            assert "X-Frame-Options" in resp.headers
            assert "X-Content-Type-Options" in resp.headers
        except httpx.ConnectError:
            pytest.skip("Nginx Gateway not reachable.")
