import pytest
from unittest.mock import MagicMock, patch
from src.shared.security import OPAEnforcer, MTLSVerifier, verify_mtls, opa_authorize
from fastapi import Request, HTTPException, status
import requests
import json

@pytest.fixture
def mock_opa_enforcer_dependencies():
    with patch('src.shared.security.requests') as mock_requests, \
         patch('src.shared.security.logger') as mock_logger:
        yield mock_requests, mock_logger

@pytest.fixture
def mock_mtls_verifier_dependencies():
    with patch('src.shared.security.logger') as mock_logger:
        yield mock_logger

# --- OPAEnforcer Tests ---
def test_opa_enforcer_init():
    enforcer = OPAEnforcer()
    assert enforcer.opa_url == "http://localhost:8181/v1/data/authz/allow"

def test_opa_enforcer_init_custom_url():
    custom_url = "http://my-opa:8181/v1/data/custom"
    enforcer = OPAEnforcer(opa_url=custom_url)
    assert enforcer.opa_url == custom_url

def test_is_authorized_success(mock_opa_enforcer_dependencies):
    mock_requests, mock_logger = mock_opa_enforcer_dependencies
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": True}
    mock_requests.post.return_value = mock_response
    
    enforcer = OPAEnforcer()
    user = {"id": "test_user", "role": "admin"}
    action = "read"
    resource = "data"
    
    assert enforcer.is_authorized(user, action, resource) is True
    mock_requests.post.assert_called_once()
    mock_logger.info.assert_called_with("opa_authorization_check", user="test_user", action=action, resource=resource, authorized=True)

def test_is_authorized_failure(mock_opa_enforcer_dependencies):
    mock_requests, mock_logger = mock_opa_enforcer_dependencies
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": False}
    mock_requests.post.return_value = mock_response
    
    enforcer = OPAEnforcer()
    user = {"id": "test_user", "role": "viewer"}
    action = "write"
    resource = "data"
    
    assert enforcer.is_authorized(user, action, resource) is False
    mock_logger.info.assert_called_with("opa_authorization_check", user="test_user", action=action, resource=resource, authorized=False)

def test_is_authorized_opa_error(mock_opa_enforcer_dependencies):
    mock_requests, mock_logger = mock_opa_enforcer_dependencies
    
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_requests.post.return_value = mock_response
    
    enforcer = OPAEnforcer()
    user = {"id": "test_user", "role": "admin"}
    action = "read"
    resource = "data"
    
    assert enforcer.is_authorized(user, action, resource) is False
    mock_logger.error.assert_called_with("opa_error", status_code=500)

def test_is_authorized_connection_error(mock_opa_enforcer_dependencies):
    mock_requests, mock_logger = mock_opa_enforcer_dependencies
    mock_requests.post.side_effect = requests.exceptions.RequestException("Connection refused")
    
    enforcer = OPAEnforcer()
    user = {"id": "test_user", "role": "admin"}
    action = "read"
    resource = "data"
    
    assert enforcer.is_authorized(user, action, resource) is False
    mock_logger.error.assert_called_with("opa_connection_failed", error="Connection refused")

# --- MTLSVerifier Tests ---
def test_mtls_verifier_init_no_required_dn():
    verifier = MTLSVerifier()
    assert verifier.required_dn is None

def test_mtls_verifier_init_with_required_dn():
    required_dn = "CN=test.client"
    verifier = MTLSVerifier(required_dn=required_dn)
    assert verifier.required_dn == required_dn

def test_mtls_verify_success(mock_mtls_verifier_dependencies):
    mock_logger = mock_mtls_verifier_dependencies
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get.side_effect = lambda key: {
        "X-SSL-Client-Verify": "SUCCESS",
        "X-SSL-Client-S-DN": "CN=test.client"
    }.get(key)
    
    verifier = MTLSVerifier(required_dn="CN=test.client")
    assert verifier.verify(mock_request) is True
    mock_logger.info.assert_called_with("mtls_verified", client_dn="CN=test.client")

def test_mtls_verify_failure_status(mock_mtls_verifier_dependencies):
    mock_logger = mock_mtls_verifier_dependencies
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get.side_effect = lambda key: {
        "X-SSL-Client-Verify": "FAILED",
        "X-SSL-Client-S-DN": "CN=test.client"
    }.get(key)
    
    verifier = MTLSVerifier()
    assert verifier.verify(mock_request) is False
    mock_logger.warning.assert_called_with("mtls_verification_failed", status="FAILED")

def test_mtls_verify_failure_dn_mismatch(mock_mtls_verifier_dependencies):
    mock_logger = mock_mtls_verifier_dependencies
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get.side_effect = lambda key: {
        "X-SSL-Client-Verify": "SUCCESS",
        "X-SSL-Client-S-DN": "CN=wrong.client"
    }.get(key)
    
    verifier = MTLSVerifier(required_dn="CN=test.client")
    assert verifier.verify(mock_request) is False
    mock_logger.warning.assert_called_with("mtls_dn_mismatch", expected="CN=test.client", actual="CN=wrong.client")

# --- FastAPI Dependency Tests ---
@pytest.mark.asyncio
async def test_verify_mtls_dependency_success():
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get.side_effect = lambda key: {
        "X-SSL-Client-Verify": "SUCCESS"
    }.get(key)
    
    with patch('src.shared.security.MTLSVerifier.verify', return_value=True) as mock_verify:
        await verify_mtls(mock_request)
        mock_verify.assert_called_once_with(mock_request)

@pytest.mark.asyncio
async def test_verify_mtls_dependency_failure():
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get.side_effect = lambda key: {
        "X-SSL-Client-Verify": "FAILED"
    }.get(key)
    
    with patch('src.shared.security.MTLSVerifier.verify', return_value=False) as mock_verify:
        with pytest.raises(HTTPException) as excinfo:
            await verify_mtls(mock_request)
        assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN
        assert "mTLS verification failed" in excinfo.value.detail
        mock_verify.assert_called_once_with(mock_request)

@pytest.mark.asyncio
async def test_opa_authorize_dependency_success():
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get.side_effect = lambda key, default=None: {
        "X-User-Id": "test_user",
        "X-User-Role": "admin"
    }.get(key, default)
    
    with patch('src.shared.security.OPAEnforcer.is_authorized', return_value=True) as mock_is_authorized:
        auth_dependency = opa_authorize(action="read", resource="data")
        await auth_dependency(mock_request)
        mock_is_authorized.assert_called_once_with({"id": "test_user", "role": "admin"}, "read", "data")

@pytest.mark.asyncio
async def test_opa_authorize_dependency_failure():
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get.side_effect = lambda key, default=None: {
        "X-User-Id": "test_user",
        "X-User-Role": "guest"
    }.get(key, default)
    
    with patch('src.shared.security.OPAEnforcer.is_authorized', return_value=False) as mock_is_authorized:
        with pytest.raises(HTTPException) as excinfo:
            auth_dependency = opa_authorize(action="write", resource="data")
            await auth_dependency(mock_request)
        assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN
        assert "OPA Authorization failed for write on data" in excinfo.value.detail
        mock_is_authorized.assert_called_once_with({"id": "test_user", "role": "guest"}, "write", "data")
