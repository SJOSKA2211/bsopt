import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_get_auth_me_returns_user_profile_with_valid_token():
    # First login with valid credentials to get a token
    login_url = f"{BASE_URL}/api/v1/auth/login"
    login_payload = {
        "email": "dev@example.com",
        "password": "password"
    }
    try:
        login_response = requests.post(login_url, json=login_payload, timeout=TIMEOUT)
        assert login_response.status_code == 200, f"Login failed with status code {login_response.status_code}"
        login_data = login_response.json()
        assert "access_token" in login_data, "access_token missing in login response"
        assert login_data.get("token_type") == "bearer", f"Unexpected token_type: {login_data.get('token_type')}"
        access_token = login_data["access_token"]

        # Use the token to call /api/v1/auth/me
        auth_me_url = f"{BASE_URL}/api/v1/auth/me"
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        auth_me_response = requests.get(auth_me_url, headers=headers, timeout=TIMEOUT)
        assert auth_me_response.status_code == 200, f"/api/v1/auth/me failed with status code {auth_me_response.status_code}"
        user_profile = auth_me_response.json()
        assert isinstance(user_profile, dict), "User profile response is not a JSON object"
        # Basic user profile fields check
        # Assuming at least email and full_name fields are present
        assert "email" in user_profile, "User profile missing 'email' field"
        assert "full_name" in user_profile or "name" in user_profile, "User profile missing 'full_name' or 'name' field"
    except requests.RequestException as e:
        assert False, f"Request failed: {e}"

test_get_auth_me_returns_user_profile_with_valid_token()