import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_post_pricing_calculate_returns_option_price_and_greeks():
    # Step 1: Login to obtain access token
    login_url = f"{BASE_URL}/api/v1/auth/login"
    login_payload = {
        "email": "dev@example.com",
        "password": "password"
    }
    try:
        login_response = requests.post(login_url, json=login_payload, timeout=TIMEOUT)
        assert login_response.status_code == 200, f"Login failed with status {login_response.status_code}"
        login_data = login_response.json()
        access_token = login_data.get("access_token")
        token_type = login_data.get("token_type")
        assert access_token and token_type == "bearer", "Access token or token type not in login response"
    except Exception as e:
        raise AssertionError(f"Authentication step failed: {e}")

    # Step 2: POST to /api/v1/pricing/calculate with valid pricing parameters and Authorization header
    pricing_url = f"{BASE_URL}/api/v1/pricing/calculate"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
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
        pricing_response = requests.post(pricing_url, json=pricing_payload, headers=headers, timeout=TIMEOUT)
        assert pricing_response.status_code == 200, f"Pricing calculate failed with status {pricing_response.status_code}"
        pricing_data = pricing_response.json()
        # Validate presence of price as number and greeks as dict with some keys
        price = pricing_data.get("price")
        greeks = pricing_data.get("greeks")

        assert isinstance(price, (int, float)), "Price should be a number"
        assert isinstance(greeks, dict), "Greeks should be a dictionary"
        # Check some common greeks keys are present
        greeks_keys = ["delta", "gamma", "vega", "theta", "rho"]
        assert any(key in greeks for key in greeks_keys), f"Greeks object missing expected keys: {greeks_keys}"
    except Exception as e:
        raise AssertionError(f"Pricing calculation step failed: {e}")

test_post_pricing_calculate_returns_option_price_and_greeks()