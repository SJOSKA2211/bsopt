import requests

def test_get_auth_me_returns_user_profile_with_valid_token():
    base_url = "http://127.0.0.1:8000"
    register_url = f"{base_url}/api/v1/auth/register"
    login_url = f"{base_url}/api/v1/auth/login"
    auth_me_url = f"{base_url}/api/v1/auth/me"
    timeout = 30

    # Test user data
    test_user = {
        "email": "testuser_tc006@example.com",
        "password": "TestPass123!",
        "full_name": "Test User TC006"
    }

    access_token = None

    # Register the user
    try:
        reg_resp = requests.post(register_url, json=test_user, timeout=timeout)
        if reg_resp.status_code not in (201, 409):
            reg_resp.raise_for_status()
        # Login user
        login_resp = requests.post(login_url, json={"email": test_user["email"], "password": test_user["password"]}, timeout=timeout)
        login_resp.raise_for_status()
        login_data = login_resp.json()
        assert "access_token" in login_data, "access_token missing in login response"
        assert login_data.get("token_type") == "bearer", "token_type is not bearer"
        access_token = login_data["access_token"]

        headers = {"Authorization": f"Bearer {access_token}"}
        auth_me_resp = requests.get(auth_me_url, headers=headers, timeout=timeout)
        auth_me_resp.raise_for_status()
        user_profile = auth_me_resp.json()
        assert auth_me_resp.status_code == 200
        assert isinstance(user_profile, dict), "User profile response is not a dict"
        assert user_profile.get("email") == test_user["email"], "Returned user email does not match"
        assert "full_name" in user_profile, "User profile missing full_name"
    finally:
        # Cleanup user by login token cannot delete user here - no delete endpoint given
        # So no deletion is performed
        pass

test_get_auth_me_returns_user_profile_with_valid_token()