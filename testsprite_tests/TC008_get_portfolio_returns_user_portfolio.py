import requests

BASE_URL = "http://127.0.0.1:8000"
LOGIN_ENDPOINT = "/api/v1/auth/login"
PORTFOLIO_ENDPOINT = "/api/v1/portfolio"

EMAIL = "dev@example.com"
PASSWORD = "password"

def test_get_portfolio_returns_user_portfolio():
    login_url = f"{BASE_URL}{LOGIN_ENDPOINT}"
    portfolio_url = f"{BASE_URL}{PORTFOLIO_ENDPOINT}"
    try:
        # Login to get access token
        login_payload = {"email": EMAIL, "password": PASSWORD}
        login_resp = requests.post(login_url, json=login_payload, timeout=30)
        assert login_resp.status_code == 200, f"Login failed with status {login_resp.status_code}"
        login_data = login_resp.json()
        assert "access_token" in login_data, "access_token missing in login response"
        assert login_data.get("token_type") == "bearer", "token_type is not bearer"

        access_token = login_data["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Get user portfolio
        portfolio_resp = requests.get(portfolio_url, headers=headers, timeout=30)
        assert portfolio_resp.status_code == 200, f"Portfolio request failed with status {portfolio_resp.status_code}"
        portfolio_data = portfolio_resp.json()
        assert isinstance(portfolio_data, dict), "Portfolio response is not a JSON object"

    except requests.RequestException as e:
        assert False, f"Request failed: {e}"

test_get_portfolio_returns_user_portfolio()