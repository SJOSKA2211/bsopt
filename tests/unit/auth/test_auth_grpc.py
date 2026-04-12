from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import grpc
import pytest

from src.auth.core.tokens import TokenData
from src.auth.grpc_server import AuthServicer
from src.shared.protos import auth_pb2


@pytest.fixture
def servicer():
    return AuthServicer()

@pytest.fixture
def mock_token_data():
    return TokenData(
        user_id="test-user-id",
        email="test@example.com",
        tier="pro",
        token_type="access",
        exp=datetime.now(UTC),
        iat=datetime.now(UTC),
        jti="test-jti",
        scopes=["read", "write"]
    )

@pytest.mark.asyncio
async def test_validate_token_success(servicer, mock_token_data):
    with patch("src.auth.grpc_server.auth_service") as mock_auth:
        mock_auth.validate_token = AsyncMock(return_value=mock_token_data)
        
        request = auth_pb2.TokenRequest(token="valid-token")
        context = AsyncMock()
        
        response = await servicer.ValidateToken(request, context)
        
        assert response.valid is True
        assert response.user_id == "test-user-id"
        assert response.email == "test@example.com"
        assert response.tier == "pro"
        assert "read" in response.scopes

@pytest.mark.asyncio
async def test_validate_token_invalid(servicer):
    with patch("src.auth.grpc_server.auth_service") as mock_auth:
        from fastapi import HTTPException
        mock_auth.validate_token = AsyncMock(side_effect=HTTPException(status_code=401, detail="Invalid token"))
        
        request = auth_pb2.TokenRequest(token="invalid-token")
        context = AsyncMock()
        
        response = await servicer.ValidateToken(request, context)
        
        assert response.valid is False
        assert response.user_id == ""

@pytest.mark.asyncio
async def test_validate_token_exception(servicer):
    with patch("src.auth.grpc_server.auth_service") as mock_auth:
        mock_auth.validate_token = AsyncMock(side_effect=Exception("Database error"))
        
        request = auth_pb2.TokenRequest(token="any-token")
        context = AsyncMock()
        
        response = await servicer.ValidateToken(request, context)
        
        assert response.valid is False
        context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)
