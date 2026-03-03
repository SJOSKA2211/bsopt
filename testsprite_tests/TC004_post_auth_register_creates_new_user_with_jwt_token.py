import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"
REGISTER_ENDPOINT = "/api/v1/auth/register"
TIMEOUT = 30


def test_post_auth_register_creates_new_user_with_jwt_token():
    unique_email = f"testuser_{uuid.uuid4()}@example.com"
    payload = {
        "email": unique_email,
        "password": "P@ssw0rd123!",
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
            timeout=TIMEOUT,
        )
        assert response.status_code == 201, f"Expected status code 201, got {response.status_code}"
        json_response = response.json()
        # Expecting the response to contain at least a "token" or similar JWT token field in user object
        assert isinstance(json_response, dict), "Response is not a JSON object"
        # Check for token presence (common naming: "token" or "access_token" or "jwt")
        token_found = False
        for key in json_response.keys():
            if "token" in key.lower():
                token_found = True
                # Basic check: token is a non-empty string
                token_value = json_response[key]
                assert isinstance(token_value, str) and len(token_value) > 10, "JWT token missing or invalid"
                break
        assert token_found, "JWT token not found in the user object"
    finally:
        # Cleanup: If user creation succeeded, attempt to delete the created user
        # This endpoint or mechanism is not described in the PRD.
        # Since deleting users isn't defined, we can't clean up here.
        # So no cleanup code is added.
        pass


test_post_auth_register_creates_new_user_with_jwt_token()