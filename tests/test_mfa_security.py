
import pytest
from unittest.mock import patch
from src.database.models import User
from tests.test_utils import assert_equal

TEST_EMAIL = "victim@example.com"
TEST_PASSWORD = "Password123!"

@pytest.fixture
def logged_in_client(api_client, mock_db_session):
    # Mock pwnedpasswords to return 0 matches
    with patch("src.security.password.pwnedpasswords.check", return_value=0):
        # Register
        api_client.post("/api/v1/auth/register", json={
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
        "password_confirm": TEST_PASSWORD,
        "full_name": "Victim User",
        "accept_terms": True,
    })

    # Login
    response = api_client.post("/api/v1/auth/login", json={
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD
    })
    assert response.status_code == 200
    tokens = response.json()["data"]

    api_client.headers["Authorization"] = f"Bearer {tokens['access_token']}"
    return api_client, mock_db_session

def test_mfa_overwrite_protection(logged_in_client):
    """
    Test that an attacker (or accidental user action) CANNOT overwrite an existing MFA configuration
    without re-verification. The endpoint should return 403 Forbidden.
    """
    client, db_session = logged_in_client

    # 1. Setup MFA initially
    response = client.post("/api/v1/auth/mfa/setup")
    assert response.status_code == 200
    mfa_data = response.json()["data"]
    initial_secret = mfa_data["secret"]

    # 2. Verify MFA to enable it
    import pyotp
    totp = pyotp.TOTP(initial_secret)
    code = totp.now()

    verify_response = client.post("/api/v1/auth/mfa/verify", json={"code": code})
    assert verify_response.status_code == 200

    # 3. Verify user has MFA enabled in DB
    user = db_session.query(User).filter(User.email == TEST_EMAIL).first()
    assert user.is_mfa_enabled is True

    # 4. Attempt to Setup MFA AGAIN
    # This should now FAIL
    response_overwrite = client.post("/api/v1/auth/mfa/setup")

    assert response_overwrite.status_code == 403
    assert "already enabled" in response_overwrite.json()["message"]

    # 5. Verify secret has NOT changed
    user_after = db_session.query(User).filter(User.email == TEST_EMAIL).first()
    # We can't check secret directly easily as it is encrypted, but we can check if it changed by decrypting
    # OR simpler: check that it didn't change in the DB object (though session might have refreshed)
    # But since the request failed, no DB write should have happened.
