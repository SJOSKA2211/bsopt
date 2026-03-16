from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.middleware.security import (
    CSRFMiddleware,
    InputSanitizationMiddleware,
    IPBlockMiddleware,
    JWTAuthenticationMiddleware,
    SecurityHeadersMiddleware,
)

app = FastAPI()


@app.get("/test_route_path")
async def route_get():
    return {"message": "success"}


@app.post("/test_route_path")
async def route_post():
    return {"message": "posted"}


@app.get("/api/v1/auth/login")
async def route_auth():
    return {"message": "login"}


# Add middlewares in order: Security Headers -> Sanitization -> Auth -> IP Block
# (Note: Starlette executes them in REVERSE order of addition for the request)
# We want IP Block to run FIRST, so add it LAST.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(InputSanitizationMiddleware)
app.add_middleware(JWTAuthenticationMiddleware)
app.add_middleware(IPBlockMiddleware, blocked_ips={"1.2.3.4"})

client = TestClient(app)


def test_security_headers():
    response = client.get("/api/v1/auth/login")
    assert response.status_code == 200
    assert "X-Frame-Options" in response.headers
    assert "Content-Security-Policy" in response.headers
    assert "X-Content-Type-Options" in response.headers
    # Cache control for sensitive paths
    assert "no-store" in response.headers.get("Cache-Control", "")


def test_ip_block():
    # IPBlock runs first now.
    response = client.get("/test_route_path", headers={"X-Forwarded-For": "1.2.3.4"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Access denied"


def test_jwt_auth_missing():
    # /test_route_path is not exempt
    response = client.get("/test_route_path")
    assert response.status_code == 401


def test_jwt_auth_legacy_bypass():
    response = client.get("/test_route_path", headers={"Authorization": "Bearer legacy-token"})
    assert response.status_code == 200
    assert response.json()["message"] == "success"


@pytest.mark.asyncio
async def test_jwt_auth_verify_fail():
    with patch(
        "services.api.middleware.security.auth_registry.verify_any",
        side_effect=Exception("Invalid"),
    ):
        response = client.get("/test_route_path", headers={"Authorization": "Bearer bad-token"})
        assert response.status_code == 401


def test_input_sanitization():
    # Should log warning for suspicious pattern
    # Add auth to get past JWT middleware
    with patch("services.api.middleware.security.logger.warning") as mock_log:
        client.post(
            "/test_route_path",
            params={"q": "<script>alert(1)</script>"},
            headers={"Authorization": "Bearer legacy-token"},
        )
        mock_log.assert_called()


def test_csrf_middleware_init():
    # CSRF bypasses in testing, but we can test init and helper methods
    mid = CSRFMiddleware(app=MagicMock())
    assert mid._is_exempt(MagicMock(url=MagicMock(path="/health"))) is True
    assert mid._is_exempt(MagicMock(url=MagicMock(path="/api/v1/auth/login"))) is True

    token = mid._generate_token()
    signed = mid._sign_token(token)
    assert mid._verify_token(signed) is True
    assert mid._verify_token("garbage") is False
