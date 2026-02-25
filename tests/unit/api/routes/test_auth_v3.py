import sys
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks, Request

from src.api.routes.auth import (
    change_password,
    mfa_setup,
    mfa_verify,
    request_password_reset,
)
from src.api.schemas.auth import (
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
mock_totp.provisioning_uri.return_value = "otpauth://totp/BSOPT:rick?secret=secret"
mock_pyotp.TOTP.return_value = mock_totp
sys.modules["pyotp"] = mock_pyotp


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_bg():
    return MagicMock(spec=BackgroundTasks)


@pytest.fixture
def mock_request():
    req = MagicMock(spec=Request)
    req.client = MagicMock()
    req.client.host = "127.0.0.1"
    req.headers = {}
    return req


@pytest.mark.asyncio
async def test_mfa_setup_success(mock_db):
    mock_user = User(id=uuid4(), email="rick@c137.com", is_active=True)
    res = await mfa_setup(mock_user, mock_db)
    assert res.data.secret == "secret"
    assert "otpauth" in res.data.provisioning_uri


@pytest.mark.asyncio
async def test_mfa_verify_success(mock_db):
    data = MFAVerifyRequest(code="123456")
    mock_user = User(id=uuid4(), email="rick@c137.com", is_active=True, mfa_secret="secret")
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        res = await mfa_verify(data, mock_user, mock_db)
        assert "Successfully" in res["message"] or "MFA" in res["message"]


@pytest.mark.asyncio
async def test_change_password_success(mock_db):
    data = PasswordChangeRequest(
        current_password="old",
        new_password="NewStrongPassword123!",
        password_confirm="NewStrongPassword123!",
        new_password_confirm="NewStrongPassword123!",
    )
    mock_user = User(id=uuid4(), email="rick@c137.com", hashed_password="old_hashed")
    with patch("src.api.routes.auth.password_service") as mock_pw:
        mock_pw.verify_password.return_value = True
        mock_pw.validate_password.return_value.is_valid = True
        mock_pw.hash_password.return_value = "new_hashed"
        res = await change_password(data, mock_user, mock_db)
        assert "success" in res["message"].lower()


@pytest.mark.asyncio
async def test_password_reset_flow(mock_db, mock_bg):
    req_data = PasswordResetRequest(email="rick@c137.com")
    mock_user = User(id=uuid4(), email="rick@c137.com")
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    with patch("src.api.routes.auth.password_service") as mock_pw:
        mock_pw.generate_reset_token.return_value = "token"
        res = await request_password_reset(req_data, mock_bg, mock_db)
        assert "sent" in res["message"].lower()
