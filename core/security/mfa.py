"""
MFA Service

Handles Multi-Factor Authentication using Time-based One-Time Passwords (TOTP).
"""

import io
import logging
from base64 import b64encode

import pyotp
import qrcode
from cryptography.fernet import Fernet

from core.shared.config import settings

logger = logging.getLogger(__name__)


class MfaService:
    def __init__(self, issuer_name: str = "BSOPT Platform"):
        self.issuer_name = issuer_name
        self._fernet = None

    @property
    def fernet(self) -> Fernet:
        """Lazy initialization of Fernet for encryption."""
        if self._fernet is None:
            key = settings.MFA_ENCRYPTION_KEY
            if not key:
                if settings.is_production:
                    raise ValueError("MFA_ENCRYPTION_KEY is missing in production")

                # In development, if not provided and not derived, we should ideally
                # have a stable fallback or error out to avoid data loss on restart.
                # Settings now derives it from BETTER_AUTH_SECRET if present.
                raise ValueError(
                    "MFA_ENCRYPTION_KEY is not set. Please set BETTER_AUTH_SECRET or MFA_ENCRYPTION_KEY."
                )

            self._fernet = Fernet(key.encode())
        return self._fernet

    def encrypt_secret(self, secret: str) -> str:
        """Encrypt the MFA secret for storage."""
        return self.fernet.encrypt(secret.encode()).decode()

    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt the MFA secret for verification."""
        return self.fernet.decrypt(encrypted_secret.encode()).decode()

    def generate_secret(self) -> str:
        """Generate a new TOTP secret key."""
        return pyotp.random_base32()

    def get_provisioning_uri(self, email: str, secret: str) -> str:
        """Get the provisioning URI for QR code generation."""
        return pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name=self.issuer_name)

    def generate_qr_code(self, uri: str) -> str:
        """Generate a QR code from a provisioning URI and return as base64."""
        img = qrcode.make(uri)
        buf = io.BytesIO()
        img.save(buf)
        buf.seek(0)
        return b64encode(buf.getvalue()).decode("utf-8")

    def verify_code(self, secret: str, code: str) -> bool:
        """Verify a TOTP code against the secret (plaintext) with clock skew support."""
        if not secret or not code:
            return False
        totp = pyotp.TOTP(secret)
        # Allow 1 periodic interval (30s) of clock skew
        return totp.verify(code, valid_window=1)


# Global MFA service instance
mfa_service = MfaService()
