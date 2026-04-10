import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks, Request, Response

from api.routes.auth import (
    change_password,
    mfa_setup,
    mfa_verify,
    request_password_reset,
)
from api.schemas.auth import (
    MFAVerifyRequest,
    PasswordChangeRequest,
    PasswordResetRequest,
)
from src.database.models import User

# Surgically insert pyotp into sys.modules
mock_pyotp = MagicMock()
mock_pyotp.random_base32.return_value = "secret"
# Ensure TOTP().provisioning_uri() returns a string
mock_totp = MagicMock()
mock_totp.provisioning_uri.return_value = "otpauth://totp/BSOPT:engineer?secret=secret"
mock_pyotp.TOTP.return_value = mock_totp
sys.modules["pyotp"] = mock_pyotp


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def mock_bg():
    return MagicMock(spec=BackgroundTasks)


@pytest.fixture
def mock_response():
    return MagicMock(spec=Response)


@pytest.fixture
def mock_request():
    req = MagicMock(spec=Request)
    req.client = MagicMock()
    req.client.host = "127.0.0.1"
    req.headers = {}
    return req


@pytest.mark.asyncio
async def test_mfa_setup_success(mock_db, mock_response):
    mock_user = User(id=uuid4(), email="engineer@bsopt.com", is_active=True, mfa_secret=None)
    with patch("api.routes.auth.auth_service") as mock_auth_svc:
        mock_auth_svc.generate_mfa_secret.return_value = "plain_secret"
        mock_auth_svc.encrypt_mfa_secret.return_value = b"encrypted_secret"
        mock_auth_svc.get_totp_uri.return_value = "otpauth://totp/BSOPT:engineer@bsopt.com?secret=plain_secret&issuer=BSOPT"
        
        res = await mfa_setup(mock_response, mock_user, mock_db)
        
        assert res.data.secret == "plain_secret"
        assert "otpauth" in res.data.provisioning_uri


@pytest.mark.asyncio
async def test_mfa_verify_success(mock_db, mock_response):
    data = MFAVerifyRequest(code="123456")
    mock_user = User(id=uuid4(), email="engineer@bsopt.com", is_active=True, mfa_secret="encrypted_secret")
    with patch("api.routes.auth.auth_service") as mock_auth_svc:
        mock_auth_svc.decrypt_mfa_secret.return_value = "plain_secret"
        mock_auth_svc.verify_mfa_code.return_value = True
        
        res = await mfa_verify(data, mock_response, mock_user, mock_db)
        
        assert "successfully" in res.message.lower() or "mfa" in res.message.lower()


@pytest.mark.asyncio
async def test_change_password_success(mock_db, mock_response):
    data = PasswordChangeRequest(
        current_password="old",
        new_password="NewStrongPassword123!",
        password_confirm="NewStrongPassword123!",
    )
    mock_user = User(id=uuid4(), email="engineer@bsopt.com", hashed_password="old_hashed")
    with patch("api.routes.auth.auth_service") as mock_auth_svc:
        mock_auth_svc.verify_password.return_value = True
        mock_auth_svc.hash_password.return_value = "new_hashed"
        
        res = await change_password(data, mock_response, mock_user, mock_db)
        
        assert "success" in res.message.lower()


@pytest.mark.asyncio
async def test_password_reset_flow(mock_db, mock_bg, mock_response):
    req_data = PasswordResetRequest(email="engineer@bsopt.com")
    mock_user = User(id=uuid4(), email="engineer@bsopt.com")
    
    # 2.0 style mocks
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_user
    mock_db.execute.return_value = mock_result
    
    with patch("api.routes.auth.auth_service") as mock_auth_svc:
        mock_auth_svc.generate_reset_token.return_value = "token"
        
        res = await request_password_reset(req_data, mock_bg, mock_response, mock_db)
        
        assert "sent" in res.message.lower()