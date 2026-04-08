import os
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.security import (
    CSRFMiddleware,
    InputSanitizationMiddleware,
    IPBlockMiddleware,
    SecurityHeadersMiddleware,
)


@pytest.fixture(autouse=True)
def enable_csrf_testing(monkeypatch):
    """Ensure CSRF is enabled for these security middleware tests."""
    monkeypatch.setenv("BSOPT_CSRF_DISABLED", "false")
    monkeypatch.setenv("TESTING", "false") # Also disable standard TESTING bypass just in case


def create_app_with_middleware(middleware_class, **kwargs):
    app = FastAPI()
    app.add_middleware(middleware_class, **kwargs)

    @app.get("/")
    async def index():
        return {"message": "ok"}

    @app.post("/post")
    async def post_endpoint():
        return {"message": "posted"}

    @app.get("/api/v1/auth/test")
    async def auth_test():
        return {"message": "auth"}

    return app


def test_security_headers_middleware():
    app = create_app_with_middleware(SecurityHeadersMiddleware)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers
    assert "Permissions-Policy" in response.headers

    # Test sensitive path no-cache
    response = client.get("/api/v1/auth/test")
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, private"


def test_csrf_middleware_get_success():
    app = create_app_with_middleware(CSRFMiddleware, secret_key="test_secret")
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "csrf_token" in response.cookies


def test_csrf_middleware_post_fail_no_token():
    app = create_app_with_middleware(CSRFMiddleware, secret_key="test_secret")
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/post")
    assert response.status_code == 403


@patch("api.middleware.security.settings")
def test_csrf_middleware_full_flow(mock_settings):
    mock_settings.JWT_SECRET = "test_secret"
    mock_settings.CORS_ORIGINS = ["http://testserver"]
    mock_settings.is_production = False

    app = create_app_with_middleware(CSRFMiddleware, secret_key="test_secret")
    client = TestClient(app)

    # 1. Get token
    response = client.get("/")
    csrf_cookie = response.cookies["csrf_token"]

    # 2. Post with token
    response = client.post(
        "/post",
        cookies={"csrf_token": csrf_cookie},
        headers={"X-CSRF-Token": csrf_cookie, "Origin": "http://testserver"},
    )
    assert response.status_code == 200


def test_ip_block_middleware():
    # Include 'testserver' in trusted proxies so X-Real-IP is respected
    app = create_app_with_middleware(IPBlockMiddleware, blocked_ips={"1.2.3.4"}, trusted_proxies={"testclient", "testserver", "127.0.0.1"})
    client = TestClient(app, raise_server_exceptions=False)

    # Allowed IP
    response = client.get("/", headers={"X-Real-IP": "5.6.7.8"})
    assert response.status_code == 200

    # Blocked IP
    response = client.get("/", headers={"X-Real-IP": "1.2.3.4"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Access denied"}


def test_ip_block_middleware_x_forwarded_for():
    app = create_app_with_middleware(IPBlockMiddleware, blocked_ips={"10.0.0.1"}, trusted_proxies={"testclient", "testserver", "127.0.0.1"})
    client = TestClient(app)

    # Test X-Forwarded-For
    response = client.get("/", headers={"X-Forwarded-For": "10.0.0.1, 192.168.1.1"})
    assert response.status_code == 403


def test_csrf_middleware_failures():
    app = create_app_with_middleware(CSRFMiddleware, secret_key="test")
    client = TestClient(app)

    # 1. Invalid Origin
    response = client.post("/post", headers={"Origin": "http://evil.com"})
    assert response.status_code == 403
    assert "Origin validation failed" in response.json()["detail"]

    # 2. Missing Token
    response = client.post("/post", headers={"Origin": "http://testserver"})
    assert response.status_code == 403
    assert "CSRF token missing" in response.json()["detail"]

    # 3. Invalid Token (Cookie)
    response = client.post(
        "/post",
        cookies={"csrf_token": "invalid"},
        headers={"Origin": "http://testserver", "X-CSRF-Token": "invalid"},
    )
    assert response.status_code == 403
    assert "Invalid CSRF token" in response.json()["detail"]

    # 4. Token Mismatch
    # Get valid token first
    resp = client.get("/")
    valid_token = resp.cookies["csrf_token"]

    response = client.post(
        "/post",
        cookies={"csrf_token": valid_token},
        headers={"Origin": "http://testserver", "X-CSRF-Token": "mismatch"},
    )
    assert response.status_code == 403
    assert "CSRF token mismatch" in response.json()["detail"]


def test_ip_block_temp_block():
    middleware = IPBlockMiddleware(MagicMock(), max_failed_attempts=2, block_duration_minutes=1)
    ip = "9.9.9.9"

    assert not middleware._is_blocked(ip)
    middleware.record_failed_attempt(ip)
    assert not middleware._is_blocked(ip)
    middleware.record_failed_attempt(ip)
    assert middleware._is_blocked(ip)

    middleware.clear_failed_attempts(ip)
    assert not middleware._is_blocked(ip) # Should be false now that it's cleared


def test_security_headers_custom_policy():
    custom_policy = {"camera": ["https://example.com"]}
    app = create_app_with_middleware(SecurityHeadersMiddleware, permissions_policy=custom_policy)
    client = TestClient(app)
    response = client.get("/")
    assert 'camera=("https://example.com")' in response.headers["Permissions-Policy"]


def test_ip_block_temp_block_expiration():
    middleware = IPBlockMiddleware(MagicMock(), max_failed_attempts=1, block_duration_minutes=10)
    ip = "9.9.9.9"

    with patch("api.middleware.security.datetime") as mock_datetime:
        mock_now = datetime.now(UTC)
        mock_datetime.now.return_value = mock_now

        # Block
        middleware.record_failed_attempt(ip)
        assert middleware._is_blocked(ip)

        # Advance time by 11 mins
        mock_datetime.now.return_value = mock_now + timedelta(minutes=11)
        assert not middleware._is_blocked(ip)
        # Should have cleared the block
        assert ip not in middleware._temporary_blocks


def test_ip_block_clean_old_attempts():
    middleware = IPBlockMiddleware(MagicMock(), max_failed_attempts=5, block_duration_minutes=10)
    ip = "8.8.8.8"

    with patch("api.middleware.security.datetime") as mock_datetime:
        base_time = datetime.now(UTC)
        mock_datetime.now.return_value = base_time

        # Record attempt 20 mins ago (should be cleaned)
        middleware._failed_attempts[ip] = [base_time - timedelta(minutes=20)]

        # Record new attempt
        middleware.record_failed_attempt(ip)

        # Should only have 1 attempt (the new one)
        assert len(middleware._failed_attempts[ip]) == 1
        assert middleware._failed_attempts[ip][0] == base_time


def test_csrf_wildcard_exempt():
    class CustomCSRFMiddleware(CSRFMiddleware):
        EXEMPT_PATHS = {"/api/v1/public/*"}

    app = create_app_with_middleware(CustomCSRFMiddleware, secret_key="test")
    client = TestClient(app)

    response = client.post("/api/v1/public/something")
    assert response.status_code == 404 # Not 403


def test_csrf_no_origin_referer():
    app = create_app_with_middleware(CSRFMiddleware, secret_key="test")
    client = TestClient(app)

    resp = client.get("/")
    token = resp.cookies["csrf_token"]

    response = client.post(
        "/post",
        cookies={"csrf_token": token},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 200


def test_csrf_invalid_origin_format():
    app = create_app_with_middleware(CSRFMiddleware, secret_key="test")
    client = TestClient(app)

    response = client.post("/post", headers={"Origin": "not-a-url"})
    assert response.status_code == 403
    assert "Origin validation failed" in response.json()["detail"]


def test_input_sanitization_middleware():
    app = create_app_with_middleware(InputSanitizationMiddleware)
    client = TestClient(app)

    # Normal request
    response = client.get("/")
    assert response.status_code == 200

    # Malicious query param
    with patch("api.middleware.security.logger") as mock_logger:
        response = client.get("/?q=<script>alert(1)</script>")
        # The middleware returns 400 Bad Request
        assert response.status_code == 400
        mock_logger.warning.assert_called()

    # Malicious header
    with patch("api.middleware.security.logger") as mock_logger:
        response = client.get("/", headers={"User-Agent": "javascript:eval(1)"})
        assert response.status_code == 400
        mock_logger.warning.assert_called()
