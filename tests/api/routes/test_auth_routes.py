import asyncio
import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.auth import (
    _send_password_reset_email,
    _send_verification_email,
    _verify_mfa_code,
    get_current_active_user,
    get_current_user,
)
from src.database import get_async_db, get_db
from src.database.models import User
from src.security.auth import get_auth_service
from src.security.password import get_password_service

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_all():
    m_auth = MagicMock(); m_auth.validate_token = AsyncMock(); m_auth.invalidate_token = AsyncMock()
    m_auth.create_token_pair.return_value = MagicMock(access_token="a", refresh_token="r", token_type="b", expires_in=1)
    app.dependency_overrides[get_auth_service] = lambda: m_auth
    m_pwd = MagicMock(); m_pwd.validate_password.return_value = MagicMock(is_valid=True); m_pwd.verify_password.return_value = True
    app.dependency_overrides[get_password_service] = lambda: m_pwd
    m_db = MagicMock()
    m_res = MagicMock()
    m_res.scalar_one_or_none.return_value = None
    m_db.execute = AsyncMock(return_value=m_res)
    m_db.commit = AsyncMock()
    m_db.rollback = AsyncMock()
    m_db.refresh = AsyncMock()
    m_db.query.return_value.filter.return_value.first.return_value = None
    app.dependency_overrides[get_db] = lambda: m_db
    app.dependency_overrides[get_async_db] = lambda: m_db
    # Global patches for settings and audit
    from src.config import settings as mock_settings
    with patch("src.api.routes.auth.log_audit"), \
         patch("src.api.schemas.auth.settings", mock_settings), \
         patch("src.api.routes.auth.settings", mock_settings):
        yield m_auth, m_pwd, m_db
    app.dependency_overrides.clear()

def create_mock_user(**kwargs):
    u = MagicMock(spec=User)
    u.id = uuid.uuid4(); u.email = "t@e.com"; u.tier = "free"; u.is_verified = True; u.is_active = True
    u.is_mfa_enabled = False; u.mfa_secret = "s"; u.mfa_backup_codes = None; u.hashed_password = "h"
    u.__table__ = MagicMock(); c = MagicMock(); c.name = "id"; u.__table__.columns = [c]
    for k, v in kwargs.items(): setattr(u, k, v)
    return u

def test_login_flow(mock_all):
    m_auth, _, m_db = mock_all; u = create_mock_user()
    m_auth.authenticate_user = AsyncMock(return_value=u)
    # Success
    assert client.post("/auth/login", json={"email": "t@e.com", "password": "Password123!"}).status_code == 200
    # MFA
    u.is_mfa_enabled = True
    with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
        assert client.post("/auth/login", json={"email": "t@e.com", "password": "p", "mfa_code": "123456"}).status_code == 200
    # DB Fail
    u.is_mfa_enabled = False; m_db.commit.side_effect = Exception("f")
    assert client.post("/auth/login", json={"email": "t@e.com", "password": "p"}).status_code == 200
    m_db.rollback.assert_called()

def test_logout_flow(mock_all):
    m_auth, _, _ = mock_all; u = create_mock_user()
    app.dependency_overrides[get_current_user] = lambda: u
    # Token logic (243)
    assert client.post("/auth/logout", headers={"Authorization": "Bearer tok"}).status_code == 200
    m_auth.invalidate_token.assert_called()
    app.dependency_overrides.pop(get_current_user, None)

def test_register_flow(mock_all):
    _, _, m_db = mock_all
    with patch("src.api.routes.auth.send_transactional_email.delay"):
        payload = {"email": "n@e.com", "password": "Password123!", "password_confirm": "Password123!", "accept_terms": True}
        # 🚀 SINGULARITY: Standardized register path and response
        response = client.post("/auth/register", json=payload)
        assert response.status_code == 201
        m_db.commit.side_effect = Exception("f")
        assert client.post("/auth/register", json=payload).status_code == 500

def test_deps_exhaustive(mock_all):
    m_auth, _, m_db = mock_all; req = MagicMock(); tok = MagicMock(user_id="123")
    with patch("src.api.routes.auth.get_auth_service", return_value=m_auth):
        m_auth.validate_token.return_value = tok
        # Cache hit
        with patch("src.utils.cache.db_cache.get_user", AsyncMock(return_value={"email":"c@e.com", "id":"123", "is_active":True})):
            assert asyncio.run(get_current_user(req, token="t", db=m_db)).email == "c@e.com"
        # DB hit & set cache
        with patch("src.utils.cache.db_cache.get_user", AsyncMock(return_value=None)), patch("src.utils.cache.db_cache.set_user", AsyncMock()) as ms:
            u2 = create_mock_user(id="123")
            m_res_deps = MagicMock()
            m_res_deps.scalar_one_or_none.return_value = u2
            m_db.execute = AsyncMock(return_value=m_res_deps)
            assert asyncio.run(get_current_user(req, token="t", db=m_db)) == u2
            assert ms.called
def test_mfa_exhaustive(mock_all):
    m_auth, _, m_db = mock_all; u = create_mock_user(mfa_secret="s", is_mfa_enabled=True)
    app.dependency_overrides[get_current_active_user] = lambda: u
    with patch("src.api.routes.auth.get_auth_service", return_value=m_auth):
        m_auth.validate_token.return_value = MagicMock(user_id=str(u.id))
        with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
            assert client.post("/api/v1/auth/mfa/verify", json={"code": "123456"}, headers={"Authorization": "Bearer t"}).status_code == 200
            assert client.post("/api/v1/auth/mfa/disable", json={"code": "123456"}, headers={"Authorization": "Bearer t"}).status_code == 200
    app.dependency_overrides.pop(get_current_active_user, None)

def test_helpers_final(mock_all):
    _, _, m_db = mock_all; u = create_mock_user(mfa_secret="e", mfa_backup_codes="h1")
    with patch("src.api.routes.auth.send_transactional_email.delay") as md:
        asyncio.run(_send_verification_email("a@b.com", "t")); assert md.called
        md.reset_mock(); asyncio.run(_send_password_reset_email("a@b.com", "t")); assert md.called
    with patch("src.api.routes.auth.AES256GCM") as mf:
        mf.return_value.decrypt.side_effect = Exception("f")
        h = hashlib.sha256(b"123").hexdigest()
        assert _verify_mfa_code(create_mock_user(mfa_secret="s", mfa_backup_codes=h), "123", db=m_db) is True
