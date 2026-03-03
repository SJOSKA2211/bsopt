import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30
TEST_USER_EMAIL = "dev@example.com"
TEST_USER_PASSWORD = "password"


def test_post_pricing_calculate_returns_option_price_and_greeks():
    # Step 1: Login to get access token
    login_url = f"{BASE_URL}/api/v1/auth/login"
    login_payload = {
        "email": TEST_USER_EMAIL,
        "password": TEST_USER_PASSWORD
    }
    try:
        login_resp = requests.post(login_url, json=login_payload, timeout=TIMEOUT)
        assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
        login_data = login_resp.json()
        access_token = login_data.get("access_token")
        token_type = login_data.get("token_type")
        assert access_token and token_type and token_type.lower() == "bearer", "Invalid access token or token type"
    except requests.RequestException as e:
        assert False, f"Login request failed: {str(e)}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Step 2: POST to /api/v1/pricing/calculate with valid parameters
    pricing_url = f"{BASE_URL}/api/v1/pricing/calculate"
    pricing_payload = {
        "s": 100,
        "k": 100,
        "t": 0.5,
        "r": 0.01,
        "sigma": 0.2,
        "option_type": "call",
        "model": "black_scholes"
    }

    try:
        pricing_resp = requests.post(pricing_url, json=pricing_payload, headers=headers, timeout=TIMEOUT)
        assert pricing_resp.status_code == 200, f"Pricing calculate failed: {pricing_resp.text}"
        pricing_data = pricing_resp.json()
        assert "price" in pricing_data and isinstance(pricing_data["price"], (int, float)), "Missing or invalid price in response"
        assert "greeks" in pricing_data and isinstance(pricing_data["greeks"], dict), "Missing or invalid greeks in response"
        # Optional: validate some of the greeks keys exist
        greeks = pricing_data["greeks"]
        required_greeks_keys = {"delta", "gamma", "theta", "vega", "rho"}
        assert required_greeks_keys.issubset(greeks.keys()), f"Greeks keys missing: {required_greeks_keys - set(greeks.keys())}"
    except requests.RequestException as e:
        assert False, f"Pricing calculate request failed: {str(e)}"


test_post_pricing_calculate_returns_option_price_and_greeks()