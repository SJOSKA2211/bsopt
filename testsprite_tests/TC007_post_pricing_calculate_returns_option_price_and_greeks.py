import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_post_pricing_calculate_returns_option_price_and_greeks():
    # First, login to get an access token
    login_url = f"{BASE_URL}/api/v1/auth/login"
    login_payload = {
        "email": "dev@example.com",
        "password": "password"
    }
    try:
        login_resp = requests.post(login_url, json=login_payload, timeout=TIMEOUT)
        assert login_resp.status_code == 200, f"Login failed with status {login_resp.status_code}"
        login_data = login_resp.json()
        access_token = login_data.get("access_token")
        token_type = login_data.get("token_type")
        assert access_token and token_type == "bearer", "Missing or invalid access token"
    except requests.RequestException as e:
        assert False, f"Login request failed: {e}"

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
        response = requests.post(pricing_url, json=pricing_payload, headers=headers, timeout=TIMEOUT)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        # Validate keys in response
        assert "price" in data, "Response missing 'price'"
        assert isinstance(data["price"], (int, float)), "'price' is not a number"
        assert "greeks" in data, "Response missing 'greeks'"
        greeks = data["greeks"]
        assert isinstance(greeks, dict), "'greeks' is not an object"
        # At least one greek key expected (can check common greek keys)
        expected_greeks_keys = {"delta", "gamma", "theta", "vega", "rho"}
        assert any(key in greeks for key in expected_greeks_keys), "None of the expected greeks found in response"
    except requests.RequestException as e:
        assert False, f"Pricing calculate request failed: {e}"

test_post_pricing_calculate_returns_option_price_and_greeks()