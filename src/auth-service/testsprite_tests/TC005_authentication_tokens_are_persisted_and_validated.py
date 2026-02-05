import time

import requests

BASE_URL = "http://localhost:4000"
AUTH_LOGIN_ENDPOINT = f"{BASE_URL}/api/auth/login"
AUTH_VALIDATE_ENDPOINT = f"{BASE_URL}/api/auth/validate"
AUTH_LOGOUT_ENDPOINT = f"{BASE_URL}/api/auth/logout"
TIMEOUT = 30

def test_authentication_tokens_persisted_and_validated():
    """
    Verify that tokens and sessions produced by the better-auth handler are correctly persisted
    in PostgreSQL and validated on subsequent requests, ensuring security and session integrity.
    """
    session = requests.Session()
    try:
        # Step 1: Perform login to obtain auth tokens (simulate login)
        login_payload = {
            "username": "testuser",
            "password": "testpassword"
        }
        headers = {
            "Content-Type": "application/json"
        }
        login_resp = session.post(AUTH_LOGIN_ENDPOINT, json=login_payload, headers=headers, timeout=TIMEOUT)
        assert login_resp.status_code == 200, f"Login failed with status {login_resp.status_code}"
        login_data = login_resp.json()
        assert "token" in login_data or "access_token" in login_data, "No token found in login response"
        # Extract token (supporting common token keys)
        token = login_data.get("token") or login_data.get("access_token")
        assert isinstance(token, str) and token.strip(), "Token is empty or invalid"

        # Step 2: Use token in Authorization header for a validate request to check session persistence
        validate_headers = {
            "Authorization": f"Bearer {token}"
        }
        validate_resp = session.get(AUTH_VALIDATE_ENDPOINT, headers=validate_headers, timeout=TIMEOUT)
        assert validate_resp.status_code == 200, f"Token validation failed with status {validate_resp.status_code}"
        validate_data = validate_resp.json()
        # Check expected fields in validation response
        assert validate_data.get("valid") is True or validate_data.get("authenticated") is True, \
            "Session not validated or not authenticated"

        # Step 3: Repeat the validate request to ensure session integrity (token persistence)
        for _ in range(2):
            time.sleep(0.1)  # small delay to simulate separate requests
            repeat_resp = session.get(AUTH_VALIDATE_ENDPOINT, headers=validate_headers, timeout=TIMEOUT)
            assert repeat_resp.status_code == 200, f"Subsequent validation failed with status {repeat_resp.status_code}"
            repeat_data = repeat_resp.json()
            assert repeat_data.get("valid") is True or repeat_data.get("authenticated") is True, \
                "Subsequent session validation failed"

        # Step 4: Perform logout to invalidate session/token
        logout_resp = session.post(AUTH_LOGOUT_ENDPOINT, headers=validate_headers, timeout=TIMEOUT)
        assert logout_resp.status_code in (200, 204), f"Logout failed with status {logout_resp.status_code}"

        # Step 5: Confirm token/session is invalidated after logout
        post_logout_resp = session.get(AUTH_VALIDATE_ENDPOINT, headers=validate_headers, timeout=TIMEOUT)
        assert post_logout_resp.status_code in (401, 403), "Token/session was not invalidated after logout"

    finally:
        session.close()

test_authentication_tokens_persisted_and_validated()
