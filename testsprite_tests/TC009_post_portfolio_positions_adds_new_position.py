import requests

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

def test_post_portfolio_positions_adds_new_position():
    # Step 1: Login to get access token
    login_url = f"{BASE_URL}/api/v1/auth/login"
    login_data = {
        "email": "dev@example.com",
        "password": "password"
    }
    try:
        login_resp = requests.post(login_url, json=login_data, timeout=TIMEOUT)
        assert login_resp.status_code == 200, f"Login failed: {login_resp.status_code} {login_resp.text}"
        login_json = login_resp.json()
        access_token = login_json.get("access_token")
        token_type = login_json.get("token_type")
        assert access_token and token_type == "bearer", "Invalid token response from login"
    except requests.RequestException as e:
        assert False, f"Login request failed: {str(e)}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Step 2: POST a new position
    post_url = f"{BASE_URL}/api/v1/portfolio/positions"
    position_data = {
        "symbol": "AAPL",
        "quantity": 10,
        "entry_price": 150
    }

    # Keep track of created position id to cleanup
    position_id = None

    try:
        post_resp = requests.post(post_url, json=position_data, headers=headers, timeout=TIMEOUT)
        assert post_resp.status_code == 201, f"Expected 201 Created, got {post_resp.status_code}. Response: {post_resp.text}"
        resp_json = post_resp.json()
        # Validate that the created position contains expected fields
        assert "id" in resp_json, "Response missing 'id' field for created position"
        assert resp_json.get("symbol") == position_data["symbol"], "Symbol mismatch"
        assert resp_json.get("quantity") == position_data["quantity"], "Quantity mismatch"
        assert resp_json.get("entry_price") == position_data["entry_price"], "Entry price mismatch"
        position_id = resp_json["id"]
    except requests.RequestException as e:
        assert False, f"POST /api/v1/portfolio/positions request failed: {str(e)}"
    finally:
        # Cleanup: Delete the created position if it exists
        if position_id:
            try:
                delete_url = f"{BASE_URL}/api/v1/portfolio/positions/{position_id}"
                del_resp = requests.delete(delete_url, headers=headers, timeout=TIMEOUT)
                assert del_resp.status_code == 200, f"Failed to delete created position with id {position_id}, status: {del_resp.status_code}"
            except requests.RequestException as e:
                # Log error but don't fail test here
                print(f"Warning: Cleanup delete request failed: {str(e)}")

test_post_portfolio_positions_adds_new_position()