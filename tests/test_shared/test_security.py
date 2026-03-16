from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.shared.security import MTLSVerifier, OPAEnforcer


def test_mtls_verifier_success():
    verifier = MTLSVerifier(required_dn="CN=backend")
    request = MagicMock()
    request.headers = {
        "X-SSL-Client-Verify": "SUCCESS",
        "X-SSL-Client-S-DN": "CN=backend",
    }
    assert verifier.verify(request) is True


def test_mtls_verifier_fail_status():
    verifier = MTLSVerifier()
    request = MagicMock()
    request.headers = {
        "X-SSL-Client-Verify": "FAILED",
        "X-SSL-Client-S-DN": "CN=backend",
    }
    assert verifier.verify(request) is False


def test_mtls_verifier_fail_dn():
    verifier = MTLSVerifier(required_dn="CN=backend")
    request = MagicMock()
    request.headers = {
        "X-SSL-Client-Verify": "SUCCESS",
        "X-SSL-Client-S-DN": "CN=hacker",
    }
    assert verifier.verify(request) is False


@pytest.mark.asyncio
@patch("core.shared.security.HttpClientManager.get_client")
async def test_opa_enforcer_allow(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": True}
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    enforcer = OPAEnforcer()
    authorized = await enforcer.is_authorized({"id": "1"}, "read", "data")
    assert authorized is True


@pytest.mark.asyncio
@patch("core.shared.security.HttpClientManager.get_client")
async def test_opa_enforcer_deny(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": False}
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client

    enforcer = OPAEnforcer()
    authorized = await enforcer.is_authorized({"id": "1"}, "read", "data")
    assert authorized is False
