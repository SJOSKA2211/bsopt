import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_get_portfolio_returns_user_portfolio():
    # First, login to get a valid access token
    login_url = f"{BASE_URL}/api/v1/auth/login"
    login_payload = {
        "email": "dev@example.com",
        "password": "password"
    }
    try:
        login_response = requests.post(login_url, json=login_payload, timeout=TIMEOUT)
        assert login_response.status_code == 200, f"Login failed with status {login_response.status_code}"
        login_data = login_response.json()
        assert "access_token" in login_data, "No access_token in login response"
        token_type = login_data.get("token_type", "bearer")
        access_token = login_data["access_token"]

        # Use the token to get portfolio
        portfolio_url = f"{BASE_URL}/api/v1/portfolio"
        headers = {
            "Authorization": f"{token_type.capitalize()} {access_token}"
        }
        portfolio_response = requests.get(portfolio_url, headers=headers, timeout=TIMEOUT)
        assert portfolio_response.status_code == 200, f"Portfolio GET returned {portfolio_response.status_code}"
        portfolio_data = portfolio_response.json()
        # Validate portfolio_data is a dict (portfolio object)
        assert isinstance(portfolio_data, dict), "Portfolio response is not a JSON object"

    except requests.RequestException as e:
        assert False, f"Request to API failed: {e}"

test_get_portfolio_returns_user_portfolio()