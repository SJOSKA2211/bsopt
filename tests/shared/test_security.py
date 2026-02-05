from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.shared.security import OPAEnforcer


@pytest.fixture
def enforcer():
    return OPAEnforcer(opa_url="http://mock-opa:8181/v1/data/authz/allow")

@pytest.mark.asyncio
@patch("src.utils.http_client.HttpClientManager.get_client")
async def test_opa_authorized(mock_get_client, enforcer):
    """Verify that authorized requests return True."""
    mock_client = AsyncMock()
    mock_get_client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": True}
    mock_client.post.return_value = mock_response
    
    user = {"id": "1", "role": "admin"}
    assert await enforcer.is_authorized(user, "delete", "all") is True

@pytest.mark.asyncio
@patch("src.utils.http_client.HttpClientManager.get_client")
async def test_opa_unauthorized(mock_get_client, enforcer):
    """Verify that unauthorized requests return False."""
    mock_client = AsyncMock()
    mock_get_client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": False}
    mock_client.post.return_value = mock_response
    
    user = {"id": "2", "role": "guest"}
    assert await enforcer.is_authorized(user, "read", "secrets") is False

@pytest.mark.asyncio
@patch("src.utils.http_client.HttpClientManager.get_client")
async def test_opa_error_handling(mock_get_client, enforcer):
    """Verify that OPA errors default to False (deny)."""
    mock_client = AsyncMock()
    mock_get_client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_client.post.return_value = mock_response
    assert await enforcer.is_authorized({"role": "admin"}, "any", "any") is False
