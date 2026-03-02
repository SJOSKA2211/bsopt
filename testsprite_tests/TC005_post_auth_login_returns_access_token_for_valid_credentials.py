import requests

def test_post_auth_login_returns_access_token_for_valid_credentials():
    base_url = "http://127.0.0.1:8000"
    login_url = f"{base_url}/api/v1/auth/login"
    headers = {"Content-Type": "application/json"}
    payload = {
        "email": "dev@example.com",
        "password": "password"
    }
    try:
        response = requests.post(login_url, json=payload, headers=headers, timeout=30)
        # Assert status code 200
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        # Assert Content-Type JSON
        content_type = response.headers.get("Content-Type", "")
        assert "application/json" in content_type, f"Expected JSON response, got {content_type}"
        data = response.json()
        # Assert access_token present and token_type equal to bearer
        assert "access_token" in data, "Response missing access_token"
        assert "token_type" in data, "Response missing token_type"
        assert data["token_type"].lower() == "bearer", f"token_type expected to be 'bearer', got {data['token_type']}"
    except requests.exceptions.RequestException as e:
        # Could be connection error or timeout indicating service unavailable, which should be handled elsewhere
        assert False, f"HTTP request failed: {e}"

test_post_auth_login_returns_access_token_for_valid_credentials()
