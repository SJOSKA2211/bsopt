import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_get_portfolio_returns_user_portfolio():
    # Credentials for login - assume these are valid and exist in the system
    login_payload = {
        "email": "dev@example.com",
        "password": "password"
    }
    login_url = f"{BASE_URL}/api/v1/auth/login"
    portfolio_url = f"{BASE_URL}/api/v1/portfolio"

    try:
        # Login to obtain access token
        login_response = requests.post(login_url, json=login_payload, timeout=TIMEOUT)
        assert login_response.status_code == 200, f"Login failed with status {login_response.status_code}"
        login_data = login_response.json()
        assert "access_token" in login_data and "token_type" in login_data, "Missing access_token or token_type in login response"

        access_token = login_data["access_token"]
        auth_header = {"Authorization": f"Bearer {access_token}"}

        # Get portfolio with the access token
        portfolio_response = requests.get(portfolio_url, headers=auth_header, timeout=TIMEOUT)
        assert portfolio_response.status_code == 200, f"Expected 200 OK, got {portfolio_response.status_code}"

        portfolio_data = portfolio_response.json()
        # Basic validation that portfolio_data is a dict (portfolio object)
        assert isinstance(portfolio_data, dict), "Portfolio response is not a JSON object"

    except requests.RequestException as e:
        assert False, f"Request failed: {e}"

test_get_portfolio_returns_user_portfolio()
