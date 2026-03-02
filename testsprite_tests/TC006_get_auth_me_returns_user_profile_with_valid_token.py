import requests

BASE_URL = "http://127.0.0.1:8000"


def test_get_auth_me_returns_user_profile_with_valid_token():
    register_url = f"{BASE_URL}/api/v1/auth/register"
    login_url = f"{BASE_URL}/api/v1/auth/login"
    auth_me_url = f"{BASE_URL}/api/v1/auth/me"

    test_email = "testuser_tc006@example.com"
    test_password = "TestPassword123!"
    test_full_name = "Test User TC006"

    headers = {"Content-Type": "application/json"}

    # Register new user
    registration_payload = {
        "email": test_email,
        "password": test_password,
        "full_name": test_full_name
    }
    try:
        reg_resp = requests.post(register_url, json=registration_payload, headers=headers, timeout=30)
        if reg_resp.status_code not in (201, 409):
            reg_resp.raise_for_status()
    except Exception as e:
        raise AssertionError(f"Registration failed unexpectedly: {e}")

    # Login with registered user credentials
    login_payload = {"email": test_email, "password": test_password}
    try:
        login_resp = requests.post(login_url, json=login_payload, headers=headers, timeout=30)
        login_resp.raise_for_status()
    except Exception as e:
        raise AssertionError(f"Login failed unexpectedly: {e}")

    login_data = login_resp.json()
    assert "access_token" in login_data and login_data.get("token_type") == "bearer", \
        "Login response missing access_token or token_type 'bearer'"

    token = login_data["access_token"]
    auth_headers = {
        "Authorization": f"Bearer {token}"
    }

    # Call /api/v1/auth/me with valid token
    try:
        auth_me_resp = requests.get(auth_me_url, headers=auth_headers, timeout=30)
    except Exception as e:
        raise AssertionError(f"Request to /api/v1/auth/me failed: {e}")

    assert auth_me_resp.status_code == 200, f"Expected status 200, got {auth_me_resp.status_code}"

    user_profile = auth_me_resp.json()
    assert isinstance(user_profile, dict), "User profile response is not a JSON object"
    assert user_profile.get("email", "").lower() == test_email.lower(), "Returned user email does not match"

    # No cleanup necessary (user created can remain)


test_get_auth_me_returns_user_profile_with_valid_token()