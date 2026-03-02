import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"
REGISTER_ENDPOINT = "/api/v1/auth/register"
TIMEOUT = 30


def test_post_auth_register_creates_new_user_with_jwt_token():
    unique_email = f"testuser_{uuid.uuid4().hex}@example.com"
    payload = {
        "email": unique_email,
        "password": "StrongP@ssw0rd123!",
        "full_name": "Test User"
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = None
    try:
        response = requests.post(
            f"{BASE_URL}{REGISTER_ENDPOINT}",
            json=payload,
            headers=headers,
            timeout=TIMEOUT
        )
        assert response.status_code == 201, f"Expected 201, got {response.status_code}"
        json_data = response.json()
        assert isinstance(json_data, dict), "Response is not a JSON object"
        # Check user object contains a JWT token (likely inside 'access_token', 'token' or similar)
        # The PRD mentions user object with JWT token but no fixed key name, so we check common JWT token fields.
        token_found = False
        jwt_token_fields = ["access_token", "token", "jwt", "id_token"]
        for field in jwt_token_fields:
            if field in json_data and isinstance(json_data[field], str) and len(json_data[field]) > 0:
                token_found = True
                break
        # Alternatively, check if there is a 'token' key anywhere inside user object
        # If not found, try to find in nested keys (e.g., nested inside 'user' or similar)
        if not token_found:
            # If response contains nested 'user' dictionary
            user_obj = json_data.get("user") or json_data.get("data")
            if isinstance(user_obj, dict):
                for field in jwt_token_fields:
                    if field in user_obj and isinstance(user_obj[field], str) and len(user_obj[field]) > 0:
                        token_found = True
                        break
        assert token_found, "JWT token not found in response JSON"

    finally:
        # Cleanup: delete the user if the API provides no delete user endpoint we skip the cleanup here
        # PRD does not specify user deletion endpoint, so no delete request can be made
        # Hence no resource cleanup possible here
        pass


test_post_auth_register_creates_new_user_with_jwt_token()