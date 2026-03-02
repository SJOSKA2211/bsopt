import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_post_portfolio_positions_adds_new_position():
    # Credentials for an existing user (should be valid in test environment)
    login_url = f"{BASE_URL}/api/v1/auth/login"
    portfolio_positions_url = f"{BASE_URL}/api/v1/portfolio/positions"

    login_payload = {
        "email": "dev@example.com",
        "password": "password"
    }

    # Perform login to get access token
    login_resp = requests.post(login_url, json=login_payload, timeout=TIMEOUT)
    assert login_resp.status_code == 200, f"Login failed with status {login_resp.status_code}"
    login_data = login_resp.json()
    assert "access_token" in login_data and "token_type" in login_data
    access_token = login_data["access_token"]
    token_type = login_data["token_type"]
    assert token_type.lower() == "bearer"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Position data to post
    position_payload = {
        "symbol": "AAPL",
        "quantity": 10,
        "entry_price": 150.00
    }

    created_position = None

    try:
        # Post new portfolio position
        post_resp = requests.post(portfolio_positions_url, json=position_payload, headers=headers, timeout=TIMEOUT)
        assert post_resp.status_code == 201, f"Expected status 201, got {post_resp.status_code}"
        created_position = post_resp.json()
        # Validate created_position contains expected keys
        assert isinstance(created_position, dict)
        assert "id" in created_position or "_id" in created_position  # id field should be present
        assert created_position.get("symbol") == position_payload["symbol"]
        assert created_position.get("quantity") == position_payload["quantity"]
        assert created_position.get("entry_price") == position_payload["entry_price"]

    finally:
        # Clean up - delete the created position if created
        if created_position:
            position_id = created_position.get("id") or created_position.get("_id")
            if position_id:
                delete_url = f"{portfolio_positions_url}/{position_id}"
                try:
                    del_resp = requests.delete(delete_url, headers=headers, timeout=TIMEOUT)
                    # Allow 200 success or 404 if already deleted
                    assert del_resp.status_code in (200, 404)
                except Exception:
                    # Ignore exceptions during cleanup
                    pass

test_post_portfolio_positions_adds_new_position()