import requests

def test_post_auth_login_returns_access_token_for_valid_credentials():
    base_url = "http://127.0.0.1:8000"
    login_url = f"{base_url}/api/v1/auth/login"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "email": "dev@example.com",
        "password": "password"
    }
    try:
        response = requests.post(login_url, json=payload, headers=headers, timeout=30)
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        json_response = response.json()
        assert "access_token" in json_response, "Response JSON missing 'access_token'"
        assert "token_type" in json_response, "Response JSON missing 'token_type'"
        assert json_response["token_type"].lower() == "bearer", f"Expected token_type 'bearer', got {json_response['token_type']}"
    except requests.RequestException as e:
        assert False, f"Request to {login_url} failed: {e}"

test_post_auth_login_returns_access_token_for_valid_credentials()