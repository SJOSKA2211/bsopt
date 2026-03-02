import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"
REGISTER_ENDPOINT = "/api/v1/auth/register"
TIMEOUT = 30


def test_post_auth_register_creates_new_user_with_jwt_token():
    unique_email = f"testuser_{uuid.uuid4().hex[:8]}@example.com"
    payload = {
        "email": unique_email,
        "password": "P@ssw0rdStrong1",
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
        # Assert the response status code is 201 Created
        assert response.status_code == 201, f"Expected status 201, got {response.status_code}"

        # Assert response JSON contains user object with JWT token
        data = response.json()
        assert isinstance(data, dict), "Response JSON is not a dictionary"
        # Check for presence of JWT token; assumed fields are 'access_token' or 'token' or 'jwt'
        # Since PRD says "User object with JWT token", we check keys
        token_keys = ["access_token", "token", "jwt", "id_token"]
        has_token = any(key in data for key in token_keys)
        assert has_token, f"Response JSON does not contain a JWT token field among {token_keys}: {data}"

        # Optionally check some user info fields existence
        # Exclude token keys and expect some user info fields
        user_fields = set(data.keys()) - set(token_keys)
        assert user_fields, "Response JSON does not contain user info fields"

    finally:
        # Clean up by deleting the created user if possible
        # No authenticated endpoint for deleting user is described in PRD
        # So cleanup is not implemented here due to missing info
        pass


test_post_auth_register_creates_new_user_with_jwt_token()