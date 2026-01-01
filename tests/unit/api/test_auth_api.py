import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.database.models import User
from src.database import get_db
from src.security.auth import get_current_user, get_current_active_user
import uuid
from datetime import datetime, timezone
from fastapi import HTTPException

client = TestClient(app)

@pytest.fixture
def mock_user():
    user = User(
        id=uuid.uuid4(),
        email="auth_test@example.com",
        hashed_password="hashed_password",
        full_name="Auth Test",
        tier="free",
        is_active=True,
        is_verified=True,
        is_mfa_enabled=False,
        created_at=datetime.now(timezone.utc)
    )
    return user

def test_login_success(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.auth_service.authenticate_user", return_value=mock_user), \
         patch("src.api.routes.auth.auth_service.create_token_pair") as mock_tokens:
        mock_tokens.return_value = MagicMock(access_token="access", refresh_token="refresh", token_type="bearer", expires_in=3600)
        payload = {"email": "auth_test@example.com", "password": "password123"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["access_token"] == "access"
    app.dependency_overrides = {}

def test_login_invalid_credentials():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.auth_service.authenticate_user", return_value=None):
        payload = {"email": "wrong@example.com", "password": "wrong"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_login_unverified(mock_user):
    mock_user.is_verified = False
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.auth_service.authenticate_user", return_value=mock_user):
        payload = {"email": mock_user.email, "password": "password123"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 403
    app.dependency_overrides = {}

def test_login_inactive(mock_user):
    mock_user.is_active = False
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.auth_service.authenticate_user", return_value=mock_user):
        payload = {"email": mock_user.email, "password": "password123"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 403
    app.dependency_overrides = {}

def test_login_mfa_required(mock_user):
    mock_user.is_mfa_enabled = True
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.auth_service.authenticate_user", return_value=mock_user):
        payload = {"email": "auth_test@example.com", "password": "password123"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["requires_mfa"] is True
    app.dependency_overrides = {}

def test_login_mfa_invalid_code(mock_user):
    mock_user.is_mfa_enabled = True
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.auth_service.authenticate_user", return_value=mock_user), \
         patch("src.api.routes.auth._verify_mfa_code", return_value=False):
        payload = {"email": "auth_test@example.com", "password": "password123", "mfa_code": "123456"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_register_success():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=True)), \
         patch("src.security.password.password_service.hash_password", return_value="hashed"), \
         patch("src.api.routes.auth.idempotency_manager.check_and_set", new_callable=AsyncMock, return_value=True):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        payload = {"email": "new@example.com", "password": "StrongPassword123!", "password_confirm": "StrongPassword123!", "full_name": "New User", "accept_terms": True}
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 201
    app.dependency_overrides = {}

def test_register_idempotency_conflict():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.idempotency_manager.check_and_set", new_callable=AsyncMock, return_value=False):
        payload = {"email": "idem@example.com", "password": "StrongPassword123!", "password_confirm": "StrongPassword123!", "accept_terms": True}
        response = client.post("/api/v1/auth/register", json=payload, headers={"Idempotency-Key": "key123"})
        assert response.status_code == 409
    app.dependency_overrides = {}

def test_register_email_exists():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = MagicMock(spec=User)
    payload = {"email": "existing@example.com", "password": "StrongPassword123!", "password_confirm": "StrongPassword123!", "accept_terms": True}
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409
    app.dependency_overrides = {}

def test_verify_email_success(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    response = client.post("/api/v1/auth/verify-email", json={"token": "valid"})
    assert response.status_code == 200
    assert mock_user.is_verified is True
    app.dependency_overrides = {}

def test_verify_email_invalid_token():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = None
    response = client.post("/api/v1/auth/verify-email", json={"token": "bad"})
    assert response.status_code == 422
    app.dependency_overrides = {}

def test_refresh_token_success(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.auth_service.validate_token", new_callable=AsyncMock) as mock_val, \
         patch("src.api.routes.auth.auth_service.invalidate_token", new_callable=AsyncMock), \
         patch("src.api.routes.auth.auth_service.create_token_pair", return_value=MagicMock(access_token="new", refresh_token="new", token_type="bearer", expires_in=3600)):
        mock_val.return_value = MagicMock(user_id=str(mock_user.id), token_type="refresh")
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "old"})
        assert response.status_code == 200
    app.dependency_overrides = {}

def test_refresh_token_invalid_type():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.auth_service.validate_token", new_callable=AsyncMock) as mock_val:
        mock_val.return_value = MagicMock(token_type="access")
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "token"})
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_refresh_token_user_inactive(mock_user):
    mock_user.is_active = False
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth.auth_service.validate_token", new_callable=AsyncMock) as mock_val:
        mock_val.return_value = MagicMock(user_id=str(mock_user.id), token_type="refresh")
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "token"})
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_confirm_password_reset_success(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    with patch("src.security.password.password_service.verify_password", return_value=True), \
         patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=True)), \
         patch("src.security.password.password_service.hash_password", return_value="new"):
        payload = {"token": "tok", "new_password": "NewPassword123!", "new_password_confirm": "NewPassword123!"}
        response = client.post("/api/v1/auth/password-reset/confirm", json=payload)
        assert response.status_code == 200
    app.dependency_overrides = {}

def test_confirm_password_reset_invalid_password(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    with patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=False, errors=["Too weak"])):
        payload = {"token": "reset_token", "new_password": "WeakPassword123", "new_password_confirm": "WeakPassword123"}
        response = client.post("/api/v1/auth/password-reset/confirm", json=payload)
        assert response.status_code == 422
    app.dependency_overrides = {}

def test_change_password_success(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.password.password_service.verify_password", return_value=True), \
         patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=True)), \
         patch("src.security.password.password_service.hash_password", return_value="new"):
        payload = {"current_password": "old", "new_password": "NewPassword123!", "new_password_confirm": "NewPassword123!"}
        response = client.post("/api/v1/auth/password-change", json=payload)
        assert response.status_code == 200
    app.dependency_overrides = {}

def test_change_password_wrong_current(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.password.password_service.verify_password", return_value=False):
        payload = {"current_password": "wrong", "new_password": "NewPassword123!", "new_password_confirm": "NewPassword123!"}
        response = client.post("/api/v1/auth/password-change", json=payload)
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_mfa_setup_success(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    response = client.post("/api/v1/auth/mfa/setup")
    assert response.status_code == 200
    app.dependency_overrides = {}

def test_mfa_verify_success(mock_user):
    mock_user.mfa_secret = "secret"
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        response = client.post("/api/v1/auth/mfa/verify", json={"code": "123456"})
        assert response.status_code == 200
        assert mock_user.is_mfa_enabled is True
    app.dependency_overrides = {}

def test_mfa_verify_no_setup(mock_user):
    mock_user.mfa_secret = None
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    response = client.post("/api/v1/auth/mfa/verify", json={"code": "123456"})
    assert response.status_code == 422
    app.dependency_overrides = {}

def test_mfa_disable_success(mock_user):
    mock_user.is_mfa_enabled = True
    mock_user.mfa_secret = "secret"
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        response = client.post("/api/v1/auth/mfa/disable", json={"code": "123456"})
        assert response.status_code == 200
        assert mock_user.is_mfa_enabled is False
    app.dependency_overrides = {}

def test_mfa_disable_not_enabled(mock_user):
    mock_user.is_mfa_enabled = False
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    response = client.post("/api/v1/auth/mfa/disable", json={"code": "123456"})
    assert response.status_code == 422
    app.dependency_overrides = {}

def test_register_weak_password():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=False, errors=["Too short"])):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        payload = {"email": "new@example.com", "password": "weak", "password_confirm": "weak", "full_name": "New User", "accept_terms": True}
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 422
    app.dependency_overrides = {}

def test_refresh_token_generic_exception():
    with patch("src.api.routes.auth.auth_service.validate_token", side_effect=Exception("Something broke")):
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "token"})
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_change_password_weak_password(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.password.password_service.verify_password", return_value=True), \
         patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=False, errors=["Too weak"])):
        payload = {"current_password": "old", "new_password": "weak", "new_password_confirm": "weak"}
        response = client.post("/api/v1/auth/password-change", json=payload)
        assert response.status_code == 422
    app.dependency_overrides = {}

def test_request_password_reset_nonexistent_user():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = None
    response = client.post("/api/v1/auth/password-reset", json={"email": "notfound@example.com"})
    assert response.status_code == 200
    app.dependency_overrides = {}

def test_setup_mfa_db_error(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.commit.side_effect = Exception("DB error")
    response = client.post("/api/v1/auth/mfa/setup")
    assert response.status_code == 500
    app.dependency_overrides = {}

def test_register_db_error():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.commit.side_effect = Exception("DB error")
    with patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=True)), \
         patch("src.security.password.password_service.hash_password", return_value="hashed"), \
         patch("src.api.routes.auth.idempotency_manager.check_and_set", new_callable=AsyncMock, return_value=True):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        payload = {"email": "new@example.com", "password": "StrongPassword123!", "password_confirm": "StrongPassword123!", "full_name": "New User", "accept_terms": True}
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 500
    app.dependency_overrides = {}