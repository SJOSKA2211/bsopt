from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import grpc
import pytest

from src.auth.core.tokens import TokenData
from src.auth.grpc_client import AuthGrpcClient
from src.auth.grpc_server import AuthServicer
from src.shared.protos import auth_pb2_grpc


@pytest.fixture
async def grpc_server_port():
    # Use a dynamic port for testing
    return "50052"

@pytest.fixture
async def run_auth_server(grpc_server_port):
    server = grpc.aio.server()
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServicer(), server)
    server.add_insecure_port(f"0.0.0.0:{grpc_server_port}")
    await server.start()
    yield
    await server.stop(0)

@pytest.mark.asyncio
async def test_auth_grpc_integration(run_auth_server, grpc_server_port, monkeypatch):
    # Point client to our test server
    monkeypatch.setenv("AUTH_SERVICE_GRPC_URL", f"localhost:{grpc_server_port}")
    monkeypatch.setenv("GRPC_SECURE", "false")
    
    client = AuthGrpcClient()
    
    mock_token_data = TokenData(
        user_id="integration-user",
        email="integration@test.com",
        tier="enterprise",
        token_type="access",
        exp=datetime.now(UTC),
        iat=datetime.now(UTC),
        jti="integration-jti",
        scopes=["admin"]
    )
    
    with patch("src.auth.grpc_server.auth_service") as mock_auth:
        mock_auth.validate_token = AsyncMock(return_value=mock_token_data)
        
        response = await client.validate_token("some-token")
        
        assert response is not None
        assert response.valid is True
        assert response.user_id == "integration-user"
        assert "admin" in response.scopes
    
    await client.close()

@pytest.mark.asyncio
async def test_auth_grpc_unauthorized(run_auth_server, grpc_server_port, monkeypatch):
    monkeypatch.setenv("AUTH_SERVICE_GRPC_URL", f"localhost:{grpc_server_port}")
    monkeypatch.setenv("GRPC_SECURE", "false")
    
    client = AuthGrpcClient()
    
    with patch("src.auth.grpc_server.auth_service") as mock_auth:
        from fastapi import HTTPException
        mock_auth.validate_token = AsyncMock(side_effect=HTTPException(status_code=401, detail="Invalid token"))
        
        response = await client.validate_token("invalid-token")
        
        assert response is not None
        assert response.valid is False
    
    await client.close()
