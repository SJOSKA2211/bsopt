import pytest
from unittest.mock import patch
from src.shared.security import OPAEnforcer

@pytest.fixture
def enforcer():
    return OPAEnforcer(opa_url="http://mock-opa:8181/v1/data/authz/allow")

@patch("src.shared.security.requests.post")
def test_opa_authorized(mock_post, enforcer):
    """Verify that authorized requests return True."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"result": True}
    
    user = {"id": "1", "role": "admin"}
    assert enforcer.is_authorized(user, "delete", "all") is True

@patch("src.shared.security.requests.post")
def test_opa_unauthorized(mock_post, enforcer):
    """Verify that unauthorized requests return False."""
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"result": False}
    
    user = {"id": "2", "role": "guest"}
    assert enforcer.is_authorized(user, "read", "secrets") is False

@patch("src.shared.security.requests.post")
def test_opa_error_handling(mock_post, enforcer):
    """Verify that OPA errors default to False (deny)."""
    mock_post.return_value.status_code = 500
    assert enforcer.is_authorized({"role": "admin"}, "any", "any") is False
