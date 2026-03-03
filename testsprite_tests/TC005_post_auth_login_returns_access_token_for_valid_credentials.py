import requests


def test_post_auth_login_returns_access_token_for_valid_credentials():
    base_url = "http://127.0.0.1:8000"
    login_url = f"{base_url}/api/v1/auth/login"
    timeout = 30

    # Use valid credentials as per PRD user flows
    payload = {"email": "dev@example.com", "password": "password"}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(login_url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as e:
        assert False, f"Request to {login_url} failed with exception: {e}"

    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

    try:
        data = response.json()
    except ValueError:
        assert False, "Response is not valid JSON"

    assert "access_token" in data, "Response JSON does not contain 'access_token'"
    assert "token_type" in data, "Response JSON does not contain 'token_type'"
    assert data["token_type"].lower() == "bearer", (
        f"token_type is not 'bearer', got {data['token_type']}"
    )

    # Further optional check: Try to access /api/v1/auth/me with returned token to confirm authentication works
    me_url = f"{base_url}/api/v1/auth/me"
    auth_headers = {"Authorization": f"Bearer {data['access_token']}"}
    try:
        me_response = requests.get(me_url, headers=auth_headers, timeout=timeout)
    except requests.exceptions.RequestException as e:
        assert False, f"Request to {me_url} failed with exception: {e}"

    assert me_response.status_code == 200, (
        f"Authenticated access to /api/v1/auth/me failed with status code {me_response.status_code}"
    )


test_post_auth_login_returns_access_token_for_valid_credentials()
