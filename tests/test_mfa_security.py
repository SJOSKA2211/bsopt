import pyotp
from unittest.mock import patch
from src.database.models import User
from cryptography.fernet import Fernet
from src.config import get_settings

TEST_EMAIL = "victim@example.com"
TEST_PASSWORD = "Password123!"


def test_mfa_overwrite_vulnerability(api_client, mock_db_session):
    # Patch pwnedpasswords.check to return 0 (not pwned)
    with patch("src.security.password.pwnedpasswords.check", return_value=0):
        # 1. Register
        register_data = {
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD,
            "password_confirm": TEST_PASSWORD,
            "full_name": "Victim User",
            "accept_terms": True
        }
        resp = api_client.post("/api/v1/auth/register", json=register_data)
        # 409 if already exists from previous run
        assert resp.status_code in [201, 409]

        # 2. Login
        login_resp = api_client.post(
            "/api/v1/auth/login",
            json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
        )
        assert login_resp.status_code == 200
        token = login_resp.json()["data"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 3. Setup MFA (First time)
        setup_resp_1 = api_client.post(
            "/api/v1/auth/mfa/setup", headers=headers
        )
        assert setup_resp_1.status_code == 200
        secret_1 = setup_resp_1.json()["data"]["secret"]

        # 4. Verify MFA to enable it
        totp = pyotp.TOTP(secret_1)
        code = totp.now()
        verify_resp = api_client.post(
            "/api/v1/auth/mfa/verify", headers=headers, json={"code": code}
        )
        assert verify_resp.status_code == 200

        # Check that user has MFA enabled in DB
        user = mock_db_session.query(User).filter(
            User.email == TEST_EMAIL
        ).first()
        assert user.is_mfa_enabled is True

        # 5. Setup MFA (Second time) - This is the attack/vulnerability
        # The user (or attacker with session) requests setup again.
        setup_resp_2 = api_client.post(
            "/api/v1/auth/mfa/setup", headers=headers
        )

        # NEW BEHAVIOR: It should return 409 Conflict
        assert setup_resp_2.status_code == 409
        assert "already enabled" in setup_resp_2.json()["message"]

        # Verify DB has NOT changed
        mock_db_session.refresh(user)
        fernet = Fernet(get_settings().MFA_ENCRYPTION_KEY)
        decrypted_secret_db = fernet.decrypt(
            user.mfa_secret.encode()
        ).decode()
        assert decrypted_secret_db == secret_1
