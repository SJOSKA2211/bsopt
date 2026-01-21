import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.database import get_db
from src.security.auth import get_auth_service, get_current_active_user, get_token_from_header, get_current_user
from src.security.password import get_password_service
from src.database.models import User
from unittest.mock import MagicMock, patch, AsyncMock
import uuid
from sqlalchemy.exc import SQLAlchemyError
from src.api.exceptions import AuthenticationException, PermissionDeniedException

client = TestClient(app)

@pytest.fixture
def mock_auth_service():
    mock = MagicMock()
    app.dependency_overrides[get_auth_service] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_auth_service, None)

@pytest.fixture
def mock_password_service():
    mock = MagicMock()
    app.dependency_overrides[get_password_service] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_password_service, None)

@pytest.fixture
def mock_db():
    mock_session = MagicMock()
    # Mock common session methods to return mocks that don't trigger real DB
    mock_session.query.return_value.filter.return_value.first.return_value = None
    app.dependency_overrides[get_db] = lambda: mock_session
    yield mock_session
    app.dependency_overrides.pop(get_db, None)

# --- Login Tests ---

def test_login_success(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.id = uuid.uuid4()
    mock_user.email = "test@ex.com"
    mock_user.is_verified = True
    mock_user.is_active = True
    mock_user.is_mfa_enabled = False
    mock_user.tier = "free"
    
    mock_auth_service.authenticate_user.return_value = mock_user
    mock_token_pair = MagicMock()
    mock_token_pair.access_token = "at"
    mock_token_pair.refresh_token = "rt"
    mock_token_pair.token_type = "bearer"
    mock_token_pair.expires_in = 3600
    mock_auth_service.create_token_pair.return_value = mock_token_pair
    
    payload = {"email": "test@ex.com", "password": "password"}
    response = client.post("/api/v1/auth/login", json=payload)
    
    assert response.status_code == 200
    assert response.json()["data"]["access_token"] == "at"

def test_login_failure_invalid_credentials(mock_auth_service, mock_db):
    mock_auth_service.authenticate_user.return_value = None
    payload = {"email": "wrong@ex.com", "password": "wrong"}
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 401

def test_login_unverified_email(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.is_verified = False
    mock_auth_service.authenticate_user.return_value = mock_user
    
    payload = {"email": "test@ex.com", "password": "password"}
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 403
    assert "not verified" in response.json()["message"]

def test_login_inactive_account(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.is_verified = True
    mock_user.is_active = False
    mock_auth_service.authenticate_user.return_value = mock_user
    
    payload = {"email": "test@ex.com", "password": "password"}
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 403
    assert "deactivated" in response.json()["message"]

def test_login_mfa_required_but_missing(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.id = uuid.uuid4()
    mock_user.email = "test@ex.com"
    mock_user.is_verified = True
    mock_user.is_active = True
    mock_user.is_mfa_enabled = True
    mock_user.tier = "free"
    mock_auth_service.authenticate_user.return_value = mock_user
    
    payload = {"email": "test@ex.com", "password": "password"}
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["requires_mfa"] is True
    assert response.json()["data"]["access_token"] == ""

def test_login_mfa_invalid_code(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.is_verified = True
    mock_user.is_active = True
    mock_user.is_mfa_enabled = True
    mock_auth_service.authenticate_user.return_value = mock_user
    
    with patch("src.api.routes.auth._verify_mfa_code", return_value=False):
        payload = {"email": "test@ex.com", "password": "password", "mfa_code": "123456"}
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 401
        assert "Invalid Multi-Factor Authentication code" in response.json()["message"]

def test_login_db_commit_error(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.id = uuid.uuid4()
    mock_user.is_verified = True
    mock_user.is_active = True
    mock_user.is_mfa_enabled = False
    mock_auth_service.authenticate_user.return_value = mock_user
    mock_auth_service.create_token_pair.return_value = MagicMock(access_token="at", refresh_token="rt", token_type="bearer", expires_in=3600)
    
    mock_db.commit.side_effect = Exception("DB Error")
    
    payload = {"email": "test@ex.com", "password": "password"}
    response = client.post("/api/v1/auth/login", json=payload)
    # Should still succeed logging in, just logs error
    assert response.status_code == 200
    mock_db.rollback.assert_called()

# --- Register Tests ---

def test_register_success(mock_password_service, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    mock_val = MagicMock()
    mock_val.is_valid = True
    mock_password_service.validate_password.return_value = mock_val
    mock_password_service.hash_password.return_value = "hashed"
    mock_password_service.generate_verification_token.return_value = "token"
    
    payload = {
        "email": "new@ex.com",
        "password": "SecurePassword123!",
        "password_confirm": "SecurePassword123!",
        "full_name": "New User",
        "accept_terms": True
    }
    
    with patch("src.api.routes.auth._send_verification_email") as mock_send:
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 201
        mock_db.add.assert_called()

def test_register_conflict(mock_db):
    mock_user = MagicMock(spec=User)
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        payload = {"email": "exist@ex.com", "password": "Pwd", "password_confirm": "Pwd", "full_name": "N", "accept_terms": True}
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 409

def test_register_invalid_password(mock_password_service, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_val = MagicMock()
    mock_val.is_valid = False
    mock_val.errors = ["Too short"]
    mock_password_service.validate_password.return_value = mock_val
    
    payload = {"email": "new@ex.com", "password": "short", "password_confirm": "short", "full_name": "N", "accept_terms": True}
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 400
    assert "Password policy violation" in response.json()["message"]

def test_register_db_error(mock_password_service, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_val = MagicMock()
    mock_val.is_valid = True
    mock_password_service.validate_password.return_value = mock_val
    mock_db.commit.side_effect = Exception("DB Fail")
    
    payload = {"email": "new@ex.com", "password": "Pwd", "password_confirm": "Pwd", "full_name": "N", "accept_terms": True}
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 500
    mock_db.rollback.assert_called()

# --- Verify Email Tests ---

def test_verify_email_success(mock_db):
    mock_user = MagicMock(spec=User)
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        payload = {"token": "valid_token"}
        response = client.post("/api/v1/auth/verify-email", json=payload)
        assert response.status_code == 200
        assert mock_user.is_verified is True

def test_verify_email_invalid_token(mock_db):
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=None):
        payload = {"token": "invalid"}
        response = client.post("/api/v1/auth/verify-email", json=payload)
        assert response.status_code == 400

def test_verify_email_db_error(mock_db):
    mock_user = MagicMock(spec=User)
    mock_db.commit.side_effect = Exception("DB Fail")
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        payload = {"token": "valid"}
        response = client.post("/api/v1/auth/verify-email", json=payload)
        assert response.status_code == 500
        mock_db.rollback.assert_called()

# --- Refresh Token Tests ---

def test_refresh_token_success(mock_auth_service, mock_db):
    mock_token_data = MagicMock()
    mock_token_data.token_type = "refresh"
    mock_token_data.user_id = str(uuid.uuid4())
    mock_auth_service.validate_token = AsyncMock(return_value=mock_token_data)
    mock_auth_service.invalidate_token = AsyncMock()
    
    mock_user = MagicMock(spec=User)
    mock_user.is_active = True
    mock_user.email = "test@ex.com"
    mock_user.tier = "free"
    
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        mock_token_pair = MagicMock()
        mock_token_pair.access_token = "new_at"
        mock_token_pair.refresh_token = "new_rt"
        mock_token_pair.token_type = "bearer"
        mock_token_pair.expires_in = 3600
        mock_auth_service.create_token_pair.return_value = mock_token_pair
        
        payload = {"refresh_token": "old_rt"}
        response = client.post("/api/v1/auth/refresh", json=payload)
        assert response.status_code == 200

def test_refresh_token_invalid_type(mock_auth_service, mock_db):
    mock_token_data = MagicMock()
    mock_token_data.token_type = "access" # Wrong type
    mock_auth_service.validate_token = AsyncMock(return_value=mock_token_data)
    
    payload = {"refresh_token": "rt"}
    response = client.post("/api/v1/auth/refresh", json=payload)
    assert response.status_code == 401

def test_refresh_token_user_not_found(mock_auth_service, mock_db):
    mock_token_data = MagicMock()
    mock_token_data.token_type = "refresh"
    mock_auth_service.validate_token = AsyncMock(return_value=mock_token_data)
    
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=None):
        payload = {"refresh_token": "rt"}
        response = client.post("/api/v1/auth/refresh", json=payload)
        assert response.status_code == 401

def test_refresh_token_exception(mock_auth_service):
    mock_auth_service.validate_token = AsyncMock(side_effect=Exception("Unexpected"))
    payload = {"refresh_token": "rt"}
    response = client.post("/api/v1/auth/refresh", json=payload)
    assert response.status_code == 401

# --- Password Reset Tests ---

def test_password_reset_request_success(mock_password_service, mock_db):
    mock_user = MagicMock(spec=User)
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        payload = {"email": "test@ex.com"}
        with patch("src.api.routes.auth._send_password_reset_email"):
            response = client.post("/api/v1/auth/password-reset", json=payload)
            assert response.status_code == 200

def test_password_reset_request_db_error(mock_db):
    mock_user = MagicMock(spec=User)
    mock_db.commit.side_effect = Exception("DB Fail")
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        payload = {"email": "test@ex.com"}
        response = client.post("/api/v1/auth/password-reset", json=payload)
        assert response.status_code == 200 # Should still return success
        mock_db.rollback.assert_called()

def test_confirm_password_reset_success(mock_password_service, mock_db):
    mock_user = MagicMock(spec=User)
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        mock_val = MagicMock()
        mock_val.is_valid = True
        mock_password_service.validate_password.return_value = mock_val
        
        payload = {"token": "token", "new_password": "NewPwd", "new_password_confirm": "NewPwd"}
        response = client.post("/api/v1/auth/password-reset/confirm", json=payload)
        assert response.status_code == 200

def test_confirm_password_reset_invalid_token(mock_db):
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=None):
        payload = {"token": "invalid", "new_password": "Pwd", "new_password_confirm": "Pwd"}
        response = client.post("/api/v1/auth/password-reset/confirm", json=payload)
        assert response.status_code == 400

def test_confirm_password_reset_weak_password(mock_password_service, mock_db):
    mock_user = MagicMock(spec=User)
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        mock_val = MagicMock()
        mock_val.is_valid = False
        mock_val.errors = ["Weak"]
        mock_password_service.validate_password.return_value = mock_val
        
        payload = {"token": "token", "new_password": "weak", "new_password_confirm": "weak"}
        response = client.post("/api/v1/auth/password-reset/confirm", json=payload)
        assert response.status_code == 400

def test_confirm_password_reset_db_error(mock_password_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_db.commit.side_effect = Exception("DB Fail")
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        mock_val = MagicMock()
        mock_val.is_valid = True
        mock_password_service.validate_password.return_value = mock_val
        
        payload = {"token": "token", "new_password": "Pwd", "new_password_confirm": "Pwd"}
        response = client.post("/api/v1/auth/password-reset/confirm", json=payload)
        assert response.status_code == 500
        mock_db.rollback.assert_called()

# --- Change Password Tests ---

def test_change_password_success(mock_password_service, mock_db):
    mock_user = MagicMock(spec=User)
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    mock_password_service.verify_password.return_value = True
    mock_val = MagicMock()
    mock_val.is_valid = True
    mock_password_service.validate_password.return_value = mock_val
    
    payload = {"current_password": "Old", "new_password": "New", "new_password_confirm": "New"}
    response = client.post("/api/v1/auth/password-change", json=payload)
    assert response.status_code == 200
    
    app.dependency_overrides.pop(get_current_active_user, None)

def test_change_password_wrong_current(mock_password_service, mock_db):
    mock_user = MagicMock(spec=User)
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    mock_password_service.verify_password.return_value = False
    
    payload = {"current_password": "Wrong", "new_password": "New", "new_password_confirm": "New"}
    response = client.post("/api/v1/auth/password-change", json=payload)
    assert response.status_code == 401
    
    app.dependency_overrides.pop(get_current_active_user, None)

def test_change_password_weak_new(mock_password_service, mock_db):
    mock_user = MagicMock(spec=User)
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    mock_password_service.verify_password.return_value = True
    mock_val = MagicMock()
    mock_val.is_valid = False
    mock_val.errors = ["Weak"]
    mock_password_service.validate_password.return_value = mock_val
    
    payload = {"current_password": "Old", "new_password": "Weak", "new_password_confirm": "Weak"}
    response = client.post("/api/v1/auth/password-change", json=payload)
    assert response.status_code == 400
    
    app.dependency_overrides.pop(get_current_active_user, None)

def test_change_password_db_error(mock_password_service, mock_db):
    mock_user = MagicMock(spec=User)
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db.commit.side_effect = Exception("DB Fail")
    
    mock_password_service.verify_password.return_value = True
    mock_val = MagicMock()
    mock_val.is_valid = True
    mock_password_service.validate_password.return_value = mock_val
    
    payload = {"current_password": "Old", "new_password": "New", "new_password_confirm": "New"}
    response = client.post("/api/v1/auth/password-change", json=payload)
    assert response.status_code == 500
    mock_db.rollback.assert_called()
    
    app.dependency_overrides.pop(get_current_active_user, None)

# --- MFA Tests ---

def test_mfa_setup_db_error(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.id = uuid.uuid4()
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db.commit.side_effect = Exception("DB Fail")
    
    with patch("pyotp.random_base32", return_value="SECRET"), patch("pyotp.TOTP"):
        response = client.post("/api/v1/auth/mfa/setup")
        assert response.status_code == 500
        mock_db.rollback.assert_called()
        
    app.dependency_overrides.pop(get_current_active_user, None)

def test_mfa_verify_not_initiated(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.mfa_secret = None
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    payload = {"code": "123"}
    response = client.post("/api/v1/auth/mfa/verify", json=payload)
    assert response.status_code == 400
    assert "not been initiated" in response.json()["message"]
    
    app.dependency_overrides.pop(get_current_active_user, None)

def test_mfa_verify_invalid_code(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.mfa_secret = "secret"
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    with patch("src.api.routes.auth._verify_mfa_code", return_value=False):
        payload = {"code": "123"}
        response = client.post("/api/v1/auth/mfa/verify", json=payload)
        assert response.status_code == 401
    
    app.dependency_overrides.pop(get_current_active_user, None)

def test_mfa_verify_db_error(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.mfa_secret = "secret"
    mock_user.id = uuid.uuid4()
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db.commit.side_effect = Exception("DB Fail")
    
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        payload = {"code": "123"}
        response = client.post("/api/v1/auth/mfa/verify", json=payload)
        assert response.status_code == 500
        mock_db.rollback.assert_called()
    
    app.dependency_overrides.pop(get_current_active_user, None)

def test_mfa_disable_success(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.is_mfa_enabled = True
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        payload = {"code": "123"}
        response = client.post("/api/v1/auth/mfa/disable", json=payload)
        assert response.status_code == 200
    
    app.dependency_overrides.pop(get_current_active_user, None)

def test_mfa_disable_already_disabled(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.is_mfa_enabled = False
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    payload = {"code": "123"}
    response = client.post("/api/v1/auth/mfa/disable", json=payload)
    assert response.status_code == 401
    
    app.dependency_overrides.pop(get_current_active_user, None)

def test_mfa_disable_invalid_code(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.is_mfa_enabled = True
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    with patch("src.api.routes.auth._verify_mfa_code", return_value=False):
        payload = {"code": "123"}
        response = client.post("/api/v1/auth/mfa/disable", json=payload)
        assert response.status_code == 401
    
    app.dependency_overrides.pop(get_current_active_user, None)

def test_mfa_disable_db_error(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.is_mfa_enabled = True
    mock_user.id = uuid.uuid4()
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db.commit.side_effect = Exception("DB Fail")
    
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        payload = {"code": "123"}
        response = client.post("/api/v1/auth/mfa/disable", json=payload)
        assert response.status_code == 500
        mock_db.rollback.assert_called()
    
    app.dependency_overrides.pop(get_current_active_user, None)

# --- Logout Tests ---

def test_logout_success(mock_auth_service):
    app.dependency_overrides[get_token_from_header] = lambda: "valid_token"
    mock_user = MagicMock(spec=User)
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    mock_auth_service.invalidate_token = AsyncMock()
    
    response = client.post("/api/v1/auth/logout")
    assert response.status_code == 200
    
    app.dependency_overrides.pop(get_token_from_header, None)
    app.dependency_overrides.pop(get_current_user, None)

def test_verify_mfa_code_helper():
    from src.api.routes.auth import _verify_mfa_code
    
    mock_user = MagicMock()
    mock_user.mfa_secret = None
    assert _verify_mfa_code(mock_user, "123") is False
    
    mock_user.mfa_secret = "encrypted"
    
    # Mock encryption/decryption error
    with patch("cryptography.fernet.Fernet") as mock_fernet:
        mock_fernet.return_value.decrypt.side_effect = Exception("Decrypt fail")
        assert _verify_mfa_code(mock_user, "123") is False
    
    # Mock success TOTP
    with patch("cryptography.fernet.Fernet") as mock_fernet, patch("pyotp.TOTP") as mock_totp:
        mock_fernet.return_value.decrypt.return_value.decode.return_value = "decrypted"
        mock_totp.return_value.verify.return_value = True
        assert _verify_mfa_code(mock_user, "123") is True

    # Mock backup code
    mock_user.mfa_secret = "encrypted"
    mock_user.mfa_backup_codes = "hashed_code"
    
    with patch("cryptography.fernet.Fernet") as mock_fernet:
        mock_fernet.return_value.decrypt.side_effect = Exception("TOTP fail")
        with patch("hashlib.sha256") as mock_hash:
            mock_hash.return_value.hexdigest.return_value = "hashed_code"
            # Without DB
            assert _verify_mfa_code(mock_user, "code") is True
            # With DB
            mock_db = MagicMock()
            assert _verify_mfa_code(mock_user, "code", db=mock_db) is True
            assert mock_user.mfa_backup_codes == "" # consumed