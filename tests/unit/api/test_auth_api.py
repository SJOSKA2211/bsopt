import pytest
from unittest.mock import MagicMock, patch, AsyncMock, ANY
from fastapi.testclient import TestClient
from src.api.main import app
from src.database.models import User
from src.database import get_db
from src.security.auth import get_current_user, get_current_active_user
import uuid
from datetime import datetime, timezone
from fastapi import HTTPException

client = TestClient(app)

# Patch log_audit globally for all tests in this file
@pytest.fixture(autouse=True)
def mock_log_audit_fixture(): # Renamed to avoid conflict if `mock_log_audit` is called elsewhere
    with patch("src.security.audit.log_audit") as mock_audit:
        yield mock_audit

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

# Helper to provide mock_user to dependencies
@pytest.fixture
def override_get_current_active_user_dependency(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    yield
    app.dependency_overrides = {}


def test_login_success(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.auth.auth_service.authenticate_user", return_value=mock_user), \
         patch("src.security.auth.auth_service.create_token_pair") as mock_tokens:
        mock_tokens.return_value = MagicMock(access_token="access", refresh_token="refresh", token_type="bearer", expires_in=3600)
        payload = {"email": "auth_test@example.com", "password": "password123"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["access_token"] == "access"
    app.dependency_overrides = {}

def test_login_db_error_updating_last_login(mock_user):
    mock_user.is_verified = True
    mock_user.is_active = True
    mock_user.is_mfa_enabled = False

    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.commit.side_effect = Exception("DB error on last login update") # Simulate DB error during commit

    with patch("src.security.auth.auth_service.authenticate_user", return_value=mock_user), \
         patch("src.security.auth.auth_service.create_token_pair") as mock_tokens:
        mock_tokens.return_value = MagicMock(access_token="access", refresh_token="refresh", token_type="bearer", expires_in=3600)
        payload = {"email": mock_user.email, "password": "password123"}
        response = client.post("/api/v1/auth/login", json=payload)
        # If db.commit fails during last_login update, the exception is caught and logged, login proceeds
        assert response.status_code == 200
        assert response.json()["data"]["access_token"] == "access"
         
        mock_db.rollback.assert_called_once() # Rollback should be called if commit fails after error
    app.dependency_overrides = {}

def test_login_invalid_credentials():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.auth.auth_service.authenticate_user", return_value=None):
        payload = {"email": "wrong@example.com", "password": "wrong"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_login_unverified(mock_user):
    mock_user.is_verified = False
    mock_user.is_active = True # Ensure active for this test
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.auth.auth_service.authenticate_user", return_value=mock_user):
        payload = {"email": mock_user.email, "password": "password123"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 403
    app.dependency_overrides = {}

def test_login_inactive(mock_user):
    mock_user.is_active = False
    mock_user.is_verified = True # Ensure verified for this test
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.auth.auth_service.authenticate_user", return_value=mock_user):
        payload = {"email": mock_user.email, "password": "password123"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 403
    app.dependency_overrides = {}

def test_login_mfa_required(mock_user):
    mock_user.is_mfa_enabled = True
    mock_user.is_verified = True
    mock_user.is_active = True
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.auth.auth_service.authenticate_user", return_value=mock_user):
        payload = {"email": "auth_test@example.com", "password": "password123"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["requires_mfa"] is True
    app.dependency_overrides = {}

def test_login_mfa_invalid_code(mock_user):
    mock_user.is_mfa_enabled = True
    mock_user.is_verified = True
    mock_user.is_active = True
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.auth.auth_service.authenticate_user", return_value=mock_user), \
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
         patch("src.api.routes.auth.idempotency_manager.check_and_set", new_callable=AsyncMock, return_value=True), \
         patch("src.api.routes.auth.BackgroundTasks.add_task") as mock_add_task:
        mock_db.query.return_value.filter.return_value.first.return_value = None
        payload = {"email": "new@example.com", "password": "StrongPassword123!", "password_confirm": "StrongPassword123!", "full_name": "New User", "accept_terms": True}
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 201
        mock_add_task.assert_called_once()
        assert mock_add_task.call_args[0][0].__name__ == "_send_verification_email" # Verify correct helper called
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

def test_register_db_error():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.commit.side_effect = Exception("DB error") # Simulate DB error during user creation
    with patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=True)), \
         patch("src.security.password.password_service.hash_password", return_value="hashed"), \
         patch("src.api.routes.auth.idempotency_manager.check_and_set", new_callable=AsyncMock, return_value=True), \
         patch("src.api.routes.auth.BackgroundTasks.add_task", new_callable=MagicMock):
        mock_db.query.return_value.filter.return_value.first.return_value = None
        payload = {"email": "new@example.example.com", "password": "StrongPassword123!", "password_confirm": "StrongPassword123!", "full_name": "New User", "accept_terms": True}
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 500
        mock_db.rollback.assert_called_once()
    app.dependency_overrides = {}

def test_verify_email_success(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    response = client.post("/api/v1/auth/verify-email", json={"token": "valid"})
    assert response.status_code == 200
    assert mock_user.is_verified is True
    mock_db.commit.assert_called_once()
    app.dependency_overrides = {}

def test_verify_email_invalid_token():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = None
    response = client.post("/api/v1/auth/verify-email", json={"token": "bad"})
    assert response.status_code == 422
    app.dependency_overrides = {}

def test_verify_email_db_error(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    mock_db.commit.side_effect = Exception("DB error on verify email commit")
    response = client.post("/api/v1/auth/verify-email", json={"token": "valid"})
    assert response.status_code == 500
    mock_db.rollback.assert_called_once()
    app.dependency_overrides = {}

# Modified test_logout
def test_logout(mock_user):
    # Override get_current_user to return a mock user
    app.dependency_overrides[get_current_user] = lambda: mock_user
    mock_user.id = str(uuid.uuid4()) # Ensure mock_user has an ID for log_audit

    # Mock auth_service.invalidate_token
    with patch("src.security.auth.auth_service.invalidate_token", new_callable=AsyncMock) as mock_invalidate_token:
        response = client.post("/api/v1/auth/logout", headers={"Authorization": "Bearer some_token"})
        assert response.status_code == 200
        assert response.json()["message"] == "Successfully logged out and session invalidated"
        mock_invalidate_token.assert_called_once_with("some_token", ANY)
    app.dependency_overrides = {}


def test_refresh_token_success(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.auth.auth_service.validate_token", new_callable=AsyncMock) as mock_val, \
         patch("src.security.auth.auth_service.invalidate_token", new_callable=AsyncMock), \
         patch("src.security.auth.auth_service.create_token_pair", return_value=MagicMock(access_token="new", refresh_token="new", token_type="bearer", expires_in=3600)):
        mock_val.return_value = MagicMock(user_id=str(mock_user.id), token_type="refresh")
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "old"})
        assert response.status_code == 200
    app.dependency_overrides = {}

def test_refresh_token_invalid_type():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.auth.auth_service.validate_token", new_callable=AsyncMock) as mock_val:
        mock_val.return_value = MagicMock(token_type="access")
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "token"})
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_refresh_token_user_inactive(mock_user):
    mock_user.is_active = False
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.auth.auth_service.validate_token", new_callable=AsyncMock) as mock_val:
        mock_val.return_value = MagicMock(user_id=str(mock_user.id), token_type="refresh")
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "token"})
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_refresh_token_generic_exception():
    with patch("src.security.auth.auth_service.validate_token", side_effect=Exception("Something broke")): # Simpler patch
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "token"})
        assert response.status_code == 401
    app.dependency_overrides = {}

def test_request_password_reset_nonexistent_user():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = None
    response = client.post("/api/v1/auth/password-reset", json={"email": "notfound@example.com"})
    assert response.status_code == 200 # Always return 200 to prevent enumeration
    app.dependency_overrides = {}

def test_request_password_reset_db_error(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    mock_db.commit.side_effect = Exception("DB error on reset token save")
    with patch("src.api.routes.auth.BackgroundTasks.add_task", new_callable=MagicMock):
        response = client.post("/api/v1/auth/password-reset", json={"email": mock_user.email})
        assert response.status_code == 200 # Still returns 200, but error logged and rollback called
        mock_db.rollback.assert_called_once()
    app.dependency_overrides = {}

def test_request_password_reset_background_task_called(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    with patch("src.api.routes.auth.BackgroundTasks.add_task") as mock_add_task:
        response = client.post("/api/v1/auth/password-reset", json={"email": mock_user.email})
        assert response.status_code == 200
        mock_add_task.assert_called_once()
        assert mock_add_task.call_args[0][0].__name__ == "_send_password_reset_email"
    app.dependency_overrides = {}

def test_confirm_password_reset_success(mock_user):
    mock_user.verification_token = "reset:tok"
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    with patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=True)), \
         patch("src.security.password.password_service.hash_password", return_value="new"):
        payload = {"token": "tok", "new_password": "NewPassword123!", "new_password_confirm": "NewPassword123!"}
        response = client.post("/api/v1/auth/password-reset/confirm", json=payload)
        assert response.status_code == 200
        assert mock_user.hashed_password == "new"
        assert mock_user.verification_token is None
        mock_db.commit.assert_called_once()
    app.dependency_overrides = {}

def test_confirm_password_reset_invalid_password_validation(mock_user):
    mock_user.verification_token = "reset:tok"
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    with patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=False, errors=["Too weak"])):
        payload = {"token": "tok", "new_password": "WeakPassword123", "new_password_confirm": "WeakPassword123"}
        response = client.post("/api/v1/auth/password-reset/confirm", json=payload)
        assert response.status_code == 422
    app.dependency_overrides = {}

def test_confirm_password_reset_db_error(mock_user):
    mock_user.verification_token = "reset:tok"
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    mock_db.commit.side_effect = Exception("DB error on password reset confirm")
    with patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=True)), \
         patch("src.security.password.password_service.hash_password", return_value="new"):
        payload = {"token": "tok", "new_password": "NewPassword123!", "new_password_confirm": "NewPassword123!"}
        response = client.post("/api/v1/auth/password-reset/confirm", json=payload)
        assert response.status_code == 500
        mock_db.rollback.assert_called_once()
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_change_password_success(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.password.password_service.verify_password", return_value=True), \
         patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=True)), \
         patch("src.security.password.password_service.hash_password", return_value="new"):
        payload = {"current_password": "old", "new_password": "NewPassword123!", "new_password_confirm": "NewPassword123!"}
        response = client.post("/api/v1/auth/password-change", json=payload)
        assert response.status_code == 200
        assert mock_user.hashed_password == "new"
        mock_db.commit.assert_called_once()
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_change_password_wrong_current(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.password.password_service.verify_password", return_value=False):
        payload = {"current_password": "wrong", "new_password": "NewPassword123!", "new_password_confirm": "NewPassword123!"}
        response = client.post("/api/v1/auth/password-change", json=payload)
        assert response.status_code == 401
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_change_password_weak_password(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.security.password.password_service.verify_password", return_value=True), \
         patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=False, errors=["Too weak"])):
        payload = {"current_password": "old", "new_password": "weak", "new_password_confirm": "weak"}
        response = client.post("/api/v1/auth/password-change", json=payload)
        assert response.status_code == 422
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_change_password_db_error(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.commit.side_effect = Exception("DB error on password change")
    with patch("src.security.password.password_service.verify_password", return_value=True), \
         patch("src.security.password.password_service.validate_password", return_value=MagicMock(is_valid=True)), \
         patch("src.security.password.password_service.hash_password", return_value="new"):
        payload = {"current_password": "old", "new_password": "NewPassword123!", "new_password_confirm": "NewPassword123!"}
        response = client.post("/api/v1/auth/password-change", json=payload)
        assert response.status_code == 500
        mock_db.rollback.assert_called_once()
    app.dependency_overrides = {}


@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_mfa_setup_success(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("pyotp.random_base32", return_value="randomsecret"), \
         patch("pyotp.TOTP") as mock_totp:
        mock_totp_instance = MagicMock()
        mock_totp_instance.provisioning_uri.return_value = "otpauth://totp/BSOPT:test@example.com?secret=randomsecret&issuer=BSOPT"
        mock_totp.return_value = mock_totp_instance
        
        response = client.post("/api/v1/auth/mfa/setup")
        assert response.status_code == 200
        assert response.json()["data"]["secret"] == "randomsecret"
        assert mock_user.mfa_secret == "randomsecret"
        mock_db.commit.assert_called_once()
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_setup_mfa_db_error(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.commit.side_effect = Exception("DB error on mfa setup")
    with patch("pyotp.random_base32", return_value="randomsecret"), \
         patch("pyotp.TOTP"):
        response = client.post("/api/v1/auth/mfa/setup")
        assert response.status_code == 500
        mock_db.rollback.assert_called_once()
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_mfa_verify_success(mock_user):
    mock_user.mfa_secret = "secret"
    mock_user.is_mfa_enabled = False # Ensure it's false before verification
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        response = client.post("/api/v1/auth/mfa/verify", json={"code": "123456"}) # Use valid digit code
        assert response.status_code == 200
        assert mock_user.is_mfa_enabled is True
        mock_db.commit.assert_called_once()
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_mfa_verify_no_setup(mock_user):
    mock_user.mfa_secret = None
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    response = client.post("/api/v1/auth/mfa/verify", json={"code": "123456"})
    assert response.status_code == 422
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_mfa_verify_invalid_code(mock_user):
    mock_user.mfa_secret = "secret"
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth._verify_mfa_code", return_value=False):
        response = client.post("/api/v1/auth/mfa/verify", json={"code": "000000"}) # Use valid digit code that will fail verification
        assert response.status_code == 401 # Expect 401 from AuthenticationException
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_mfa_verify_db_error(mock_user):
    mock_user.is_mfa_enabled = False # Ensure is_mfa_enabled is False before db error
    mock_user.mfa_secret = "secret" # Set mfa_secret to allow validation to pass
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db # Keep get_db override
    mock_db.commit.side_effect = Exception("DB error on mfa verify")
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        response = client.post("/api/v1/auth/mfa/verify", json={"code": "123456"})
        assert response.status_code == 500
        mock_db.rollback.assert_called_once()
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_mfa_disable_success(mock_user):
    mock_user.is_mfa_enabled = True
    mock_user.mfa_secret = "secret"
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        response = client.post("/api/v1/auth/mfa/disable", json={"code": "123456"}) # Use valid digit code
        assert response.status_code == 200
        assert mock_user.is_mfa_enabled is False
        assert mock_user.mfa_secret is None
        mock_db.commit.assert_called_once()
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_mfa_disable_not_enabled(mock_user):
    mock_user.is_mfa_enabled = False
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    response = client.post("/api/v1/auth/mfa/disable", json={"code": "123456"})
    assert response.status_code == 422
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_disable_mfa_invalid_code(mock_user):
    mock_user.is_mfa_enabled = True
    mock_user.mfa_secret = "secret"
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    with patch("src.api.routes.auth._verify_mfa_code", return_value=False):
        response = client.post("/api/v1/auth/mfa/disable", json={"code": "000000"}) # Use valid digit code that will fail verification
        assert response.status_code == 401 # Expect 401 from AuthenticationException
    app.dependency_overrides = {}

@pytest.mark.usefixtures("override_get_current_active_user_dependency")
def test_disable_mfa_db_error(mock_user):
    mock_user.is_mfa_enabled = True
    mock_user.mfa_secret = "secret"
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.commit.side_effect = Exception("DB error on mfa disable")
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        response = client.post("/api/v1/auth/mfa/disable", json={"code": "123456"})
        assert response.status_code == 500
        mock_db.rollback.assert_called_once()
    app.dependency_overrides = {}