import pytest
from unittest.mock import patch, MagicMock
from fastapi import Request, HTTPException
from src.shared.security import OPAEnforcer, MTLSVerifier, verify_mtls, opa_authorize
import requests
import runpy

def test_opa_enforcer_authorized():
    enforcer = OPAEnforcer()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": True}
    
    with patch('requests.post', return_value=mock_response):
        assert enforcer.is_authorized({"id": "u1"}, "read", "res") is True

def test_opa_enforcer_denied():
    enforcer = OPAEnforcer()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": False}
    
    with patch('requests.post', return_value=mock_response):
        assert enforcer.is_authorized({"id": "u1"}, "read", "res") is False

def test_opa_enforcer_error_status():
    enforcer = OPAEnforcer()
    mock_response = MagicMock()
    mock_response.status_code = 500
    
    with patch('requests.post', return_value=mock_response):
        assert enforcer.is_authorized({"id": "u1"}, "read", "res") is False

def test_opa_enforcer_exception():
    enforcer = OPAEnforcer()
    with patch('requests.post', side_effect=requests.RequestException("Conn error")):
        assert enforcer.is_authorized({"id": "u1"}, "read", "res") is False

def test_mtls_verifier_success():
    verifier = MTLSVerifier()
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-SSL-Client-Verify": "SUCCESS", "X-SSL-Client-S-DN": "CN=client"}
    assert verifier.verify(mock_request) is True

def test_mtls_verifier_failed_status():
    verifier = MTLSVerifier()
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-SSL-Client-Verify": "NONE"}
    assert verifier.verify(mock_request) is False

def test_mtls_verifier_dn_mismatch():
    verifier = MTLSVerifier(required_dn="CN=expected")
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-SSL-Client-Verify": "SUCCESS", "X-SSL-Client-S-DN": "CN=wrong"}
    assert verifier.verify(mock_request) is False

def test_mtls_verifier_dn_match():
    verifier = MTLSVerifier(required_dn="CN=expected")
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-SSL-Client-Verify": "SUCCESS", "X-SSL-Client-S-DN": "CN=expected"}
    assert verifier.verify(mock_request) is True

@pytest.mark.asyncio
async def test_verify_mtls_dependency_success():
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-SSL-Client-Verify": "SUCCESS"}
    await verify_mtls(mock_request) # Should not raise

@pytest.mark.asyncio
async def test_verify_mtls_dependency_failure():
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-SSL-Client-Verify": "FAILED"}
    with pytest.raises(HTTPException) as exc:
        await verify_mtls(mock_request)
    assert exc.value.status_code == 403

@pytest.mark.asyncio
async def test_opa_authorize_dependency_success():
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-User-Id": "u1", "X-User-Role": "admin"}
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": True}
    
    auth_dep = opa_authorize("write", "data")
    with patch('requests.post', return_value=mock_response):
        await auth_dep(mock_request) # Should not raise

@pytest.mark.asyncio
async def test_opa_authorize_dependency_failure():
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-User-Id": "u1", "X-User-Role": "guest"}
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": False}
    
    auth_dep = opa_authorize("write", "data")
    with patch('requests.post', return_value=mock_response):
        with pytest.raises(HTTPException) as exc:
            await auth_dep(mock_request)
        assert exc.value.status_code == 403

def test_security_main_execution():
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": True}
        mock_post.return_value = mock_response
        
        with patch('builtins.print'):
            runpy.run_module('src.shared.security', run_name='__main__')
            mock_post.assert_called_once()
