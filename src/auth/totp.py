"""
TOTP MFA Service (DEPRECATED)

DEPRECATED: This module is marked for deletion.
All logic has been consolidated into src/auth/auth.py (AuthService).
"""

import pyotp
import structlog

logger = structlog.get_logger(__name__)


class TOTPService:
    """
    Handle TOTP generation, verification, and provisioning.
    """

    def __init__(self, issuer_name: str = "Manifold") -> None:
        self.issuer_name = issuer_name

    def generate_secret(self) -> str:
        """Generate a new base32 TOTP secret."""
        return pyotp.random_base32()

    def get_provisioning_uri(self, email: str, secret: str) -> str:
        """Generate a provisioning URI for QR codes."""
        return pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name=self.issuer_name)

    def verify_token(self, secret: str, token: str) -> bool:
        """
        Verify a TOTP token with a small window for clock skew.
        """
        if not secret or not token:
            return False

        totp = pyotp.totp.TOTP(secret)
        # Allow 1 periodic interval (30s) of clock skew
        return totp.verify(token, valid_window=1)

    def generate_current_token(self, secret: str) -> str:
        """Generate the current TOTP token (for internal testing/automation)."""
        totp = pyotp.totp.TOTP(secret)
        return totp.now()


totp_service = TOTPService()
