import unittest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from src.api.middleware.security import (
    SecurityHeadersMiddleware,
    CSRFMiddleware,
    IPBlockMiddleware,
    InputSanitizationMiddleware
)

class TestSecurityMiddleware(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        
        @self.app.get("/")
        def home():
            return {"message": "home"}
            
        @self.app.post("/submit")
        def submit():
            return {"message": "submitted"}

    def test_security_headers(self):
        self.app.add_middleware(SecurityHeadersMiddleware)
        client = TestClient(self.app, raise_server_exceptions=False)
        
        response = client.get("/")
        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in response.headers

    def test_csrf_middleware_get_token(self):
        # GET request should set CSRF cookie
        self.app.add_middleware(CSRFMiddleware, secret_key="test-secret")
        client = TestClient(self.app, raise_server_exceptions=False)
        
        response = client.get("/")
        assert response.status_code == 200
        assert "csrf_token" in response.cookies

    def test_csrf_middleware_post_without_token(self):
        self.app.add_middleware(CSRFMiddleware, secret_key="test-secret")
        client = TestClient(self.app)
        
        # POST without token should fail
        # Middleware raises HTTPException which bubbles up
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            client.post("/submit")
        self.assertEqual(cm.exception.status_code, 403)
        self.assertEqual(cm.exception.detail, "CSRF token missing")

    def test_csrf_middleware_post_with_token(self):
        self.app.add_middleware(CSRFMiddleware, secret_key="test-secret")
        client = TestClient(self.app, raise_server_exceptions=False)
        
        # 1. Get token
        resp_get = client.get("/")
        csrf_cookie_val = resp_get.cookies["csrf_token"]
        
        # 2. Post with token in header and cookie
        # Explicitly passing cookies to be safe
        response = client.post(
            "/submit", 
            headers={"X-CSRF-Token": csrf_cookie_val}, 
            cookies={"csrf_token": csrf_cookie_val}
        )
        assert response.status_code == 200

    def test_ip_block_middleware(self):
        self.app.add_middleware(IPBlockMiddleware, blocked_ips={"1.2.3.4"})
        client = TestClient(self.app)
        
        # 1. Blocked IP
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            client.get("/", headers={"X-Real-IP": "1.2.3.4"})
        self.assertEqual(cm.exception.status_code, 403)
        
        # 2. Allowed IP
        response = client.get("/", headers={"X-Real-IP": "5.6.7.8"})
        assert response.status_code == 200

    def test_input_sanitization_middleware(self):
        self.app.add_middleware(InputSanitizationMiddleware, log_suspicious=True)
        client = TestClient(self.app)
        
        # 1. Safe request
        response = client.get("/?q=hello")
        assert response.status_code == 200
        
        # 2. Suspicious request (logging only, assuming default config doesn't block)
        with patch("src.api.middleware.security.logger") as mock_logger:
            response = client.get("/?q=<script>alert(1)</script>")
            assert response.status_code == 200
            mock_logger.warning.assert_called()

if __name__ == '__main__':
    unittest.main()
