import pytest
import requests
from src.shared.security import OPAEnforcer, MTLSVerifier, verify_mtls, opa_authorize
from fastapi import Request, HTTPException, status

def test_opa_enforcer_authorized(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"result": True}
    
    enforcer = OPAEnforcer()
    assert enforcer.is_authorized({"id": "u1"}, "read", "res") is True

def test_opa_enforcer_unauthorized(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"result": False}
    
    enforcer = OPAEnforcer()
    assert enforcer.is_authorized({"id": "u1"}, "read", "res") is False

def test_opa_enforcer_error(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 500
    
    enforcer = OPAEnforcer()
    assert enforcer.is_authorized({"id": "u1"}, "read", "res") is False

def test_opa_enforcer_connection_failed(mocker):
    mocker.patch("requests.post", side_effect=requests.exceptions.ConnectionError())
    enforcer = OPAEnforcer()
    assert enforcer.is_authorized({"id": "u1"}, "read", "res") is False

def test_mtls_verifier(mocker):
    verifier = MTLSVerifier(required_dn="CN=trusted")
    
    # Success
    mock_req = mocker.Mock(spec=Request)
    mock_req.headers = {"X-SSL-Client-Verify": "SUCCESS", "X-SSL-Client-S-DN": "CN=trusted"}
    assert verifier.verify(mock_req) is True
    
    # Verify status fail
    mock_req.headers = {"X-SSL-Client-Verify": "NONE"}
    assert verifier.verify(mock_req) is False
    
    # DN mismatch
    mock_req.headers = {"X-SSL-Client-Verify": "SUCCESS", "X-SSL-Client-S-DN": "CN=attacker"}
    assert verifier.verify(mock_req) is False

@pytest.mark.asyncio
async def test_verify_mtls_dependency(mocker):
    mock_req = mocker.Mock(spec=Request)
    mock_req.headers = {"X-SSL-Client-Verify": "SUCCESS", "X-SSL-Client-S-DN": "some-dn"}
    
    await verify_mtls(mock_req) # Should not raise
    
    mock_req.headers = {"X-SSL-Client-Verify": "FAIL"}
    with pytest.raises(HTTPException) as exc:
        await verify_mtls(mock_req)
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN

@pytest.mark.asyncio
async def test_opa_authorize_dependency(mocker):
    mock_req = mocker.Mock(spec=Request)
    mock_req.headers = {"X-User-Id": "u1", "X-User-Role": "admin"}
    
    mocker.patch("src.shared.security.OPAEnforcer.is_authorized", return_value=True)
    auth_dep = opa_authorize("read", "market")
    await auth_dep(mock_req) # Should not raise
    
    mocker.patch("src.shared.security.OPAEnforcer.is_authorized", return_value=False)
    with pytest.raises(HTTPException):
        await auth_dep(mock_req)
