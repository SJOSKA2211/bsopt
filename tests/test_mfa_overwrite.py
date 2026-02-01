import pytest
import pyotp

TEST_EMAIL = "test_mfa_overwrite@example.com"
TEST_PASSWORD = "StrongPassword123!"

@pytest.fixture
def logged_in_client(api_client):
    auth_data = {
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD,
        "password_confirm": TEST_PASSWORD,
        "full_name": "Test User",
        "accept_terms": True,
    }

    # Register
    api_client.post("/api/v1/auth/register", json=auth_data)

    # Login
    response = api_client.post(
        "/api/v1/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200
    tokens = response.json()["data"]

    api_client.headers["Authorization"] = f"Bearer {tokens['access_token']}"
    return api_client

def test_mfa_setup_overwrite_fails(logged_in_client):
    """
    Test that MFA setup cannot be called again if MFA is already enabled.
    This prevents an attacker who hijacked the session from replacing the MFA secret.
    """
    client = logged_in_client

    # 1. Setup MFA (First time)
    response = client.post("/api/v1/auth/mfa/setup")
    assert response.status_code == 200
    mfa_data = response.json()["data"]
    secret = mfa_data["secret"]

    # 2. Enable MFA (Verify)
    totp = pyotp.TOTP(secret)
    code = totp.now()

    verify_response = client.post("/api/v1/auth/mfa/verify", json={"code": code})
    assert verify_response.status_code == 200

    # 3. Attempt Setup MFA AGAIN (Should fail)
    response_again = client.post("/api/v1/auth/mfa/setup")

    # Expecting 422 Unprocessable Entity (ValidationException), or 400/403
    assert response_again.status_code in [400, 403, 422], \
        f"MFA setup should fail if already enabled, but got {response_again.status_code}"
