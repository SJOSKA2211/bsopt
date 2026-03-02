import requests

BASE_URL = "http://127.0.0.1:8000"
REGISTER_URL = f"{BASE_URL}/api/v1/auth/register"
LOGIN_URL = f"{BASE_URL}/api/v1/auth/login"
POSITIONS_URL = f"{BASE_URL}/api/v1/portfolio/positions"
DELETE_POSITION_URL = f"{BASE_URL}/api/v1/portfolio/positions/{{id}}"
TIMEOUT = 30

def test_post_portfolio_positions_adds_new_position():
    # Register a new user to get fresh credentials
    register_payload = {
        "email": "testuser_tc009@example.com",
        "password": "P@ssw0rd123!",
        "full_name": "Test User TC009"
    }
    try:
        reg_resp = requests.post(REGISTER_URL, json=register_payload, timeout=TIMEOUT)
        assert reg_resp.status_code == 201, f"Registration failed: {reg_resp.text}"
    except requests.RequestException as e:
        assert False, f"Registration request failed: {e}"

    # Login to get access token
    login_payload = {
        "email": register_payload["email"],
        "password": register_payload["password"]
    }
    try:
        login_resp = requests.post(LOGIN_URL, json=login_payload, timeout=TIMEOUT)
        assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
        login_data = login_resp.json()
        access_token = login_data.get("access_token")
        token_type = login_data.get("token_type")
        assert access_token and token_type.lower() == "bearer", "Invalid token response"
    except requests.RequestException as e:
        assert False, f"Login request failed: {e}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    position_payload = {
        "symbol": "AAPL",
        "quantity": 10,
        "entry_price": 150
    }

    position_id = None
    try:
        # Post new position
        post_resp = requests.post(POSITIONS_URL, json=position_payload, headers=headers, timeout=TIMEOUT)
        assert post_resp.status_code == 201, f"Create position failed: {post_resp.text}"
        post_data = post_resp.json()
        position_id = post_data.get("id")
        assert position_id is not None, "Created position response missing 'id'"
        assert post_data.get("symbol") == position_payload["symbol"], "Position symbol mismatch"
        assert post_data.get("quantity") == position_payload["quantity"], "Position quantity mismatch"
        assert post_data.get("entry_price") == position_payload["entry_price"], "Position entry_price mismatch"
    finally:
        # Clean up: delete created position if created
        if position_id:
            try:
                del_resp = requests.delete(DELETE_POSITION_URL.format(id=position_id), headers=headers, timeout=TIMEOUT)
                assert del_resp.status_code == 200, f"Delete position failed: {del_resp.text}"
            except requests.RequestException:
                pass  # best effort cleanup

test_post_portfolio_positions_adds_new_position()