
import sys
from datetime import datetime
from uuid import uuid4
from typing import Any

# Mocking Protobuf objects
class MockTimestamp:
    def __init__(self, dt=None):
        self.dt = dt or datetime.utcnow()
    def to_datetime(self):
        return self.dt

class MockProto:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def HasField(self, field):
        return hasattr(self, field) and getattr(self, field) is not None

def test_imports():
    print("Testing imports...")
    try:
        from src.api.schemas.common import ErrorResponse, PaginationMeta
        from src.api.schemas.user import UserResponse
        from src.api.schemas.auth import TokenResponse, AuthResponse
        print("Imports successful.")
    except Exception as e:
        print(f"Import failed: {e}")
        sys.exit(1)

def test_from_proto():
    print("Testing from_proto mappings...")
    
    from src.api.schemas.common import ErrorResponse, PaginationMeta, ErrorDetail
    from src.api.schemas.user import UserResponse
    from src.api.schemas.auth import TokenResponse, AuthResponse

    # Test ErrorResponse.from_proto
    mock_error_field = MockProto(field="email", message="Invalid", code="invalid")
    mock_error_proto = MockProto(
        code="VALIDATION_ERROR",
        message="Validation failed",
        errors=[mock_error_field],
        request_id="req_123",
        timestamp=MockTimestamp()
    )
    error_resp = ErrorResponse.from_proto(mock_error_proto)
    assert error_resp.error == "VALIDATION_ERROR"
    assert error_resp.details[0].field == "email"
    print("ErrorResponse.from_proto passed.")

    # Test PaginationMeta.from_proto
    mock_pagination_proto = MockProto(
        total_items=100,
        current_page=1,
        page_size=10,
        total_pages=10,
        has_next=True,
        has_previous=False
    )
    pag_meta = PaginationMeta.from_proto(mock_pagination_proto)
    assert pag_meta.total == 100
    assert pag_meta.has_next is True
    print("PaginationMeta.from_proto passed.")

    # Test UserResponse.from_proto
    user_id = str(uuid4())
    mock_user_proto = MockProto(
        user_id=user_id,
        email="test@example.com",
        full_name="Test User",
        tier="pro",
        is_verified=True,
        mfa_enabled=False,
        created_at=MockTimestamp(),
        last_login=MockTimestamp()
    )
    user_resp = UserResponse.from_proto(mock_user_proto)
    assert str(user_resp.id) == user_id
    assert user_resp.email == "test@example.com"
    print("UserResponse.from_proto passed.")

    # Test TokenResponse.from_proto
    mock_token_proto = MockProto(
        access_token="access",
        refresh_token="refresh",
        token_type="bearer",
        expires_in=3600
    )
    token_resp = TokenResponse.from_proto(mock_token_proto)
    assert token_resp.access_token == "access"
    assert token_resp.expires_in == 3600
    print("TokenResponse.from_proto passed.")

    # Test AuthResponse.from_proto
    mock_auth_proto = MockProto(
        authenticated=True,
        user_id="user_123",
        factors_verified=["password", "mfa"]
    )
    auth_resp = AuthResponse.from_proto(mock_auth_proto)
    assert auth_resp.authenticated is True
    assert "mfa" in auth_resp.factors_verified
    print("AuthResponse.from_proto passed.")

if __name__ == "__main__":
    test_imports()
    test_from_proto()
    print("All tests passed!")
