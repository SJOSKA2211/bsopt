"""
Multi-Factor Authentication Substrate (TOTP).
"""

import logging
import pyotp
from cryptography.fernet import Fernet
from src.shared.config import settings

logger = logging.getLogger(__name__)

class MFAService:
    """
    TOTP-based MFA management with Fernet encryption for secrets.
    """
    def __init__(self):
        self._fernet = None

    @property
    def fernet(self) -> Fernet:
        """Lazy initialization of Fernet for MFA secret encryption."""
        if self._fernet is None:
            key = settings.MFA_ENCRYPTION_KEY
            if not key:
                raise ValueError("MFA_ENCRYPTION_KEY is missing")
            self._fernet = Fernet(key.encode())
        return self._fernet

    def generate_mfa_secret(self) -> str:
        """Generate a new TOTP secret."""
        return pyotp.random_base32()

    def encrypt_mfa_secret(self, secret: str) -> str:
        """Encrypt MFA secret for database storage."""
        return self.fernet.encrypt(secret.encode()).decode()

    def decrypt_mfa_secret(self, encrypted_secret: str) -> str:
        """Decrypt MFA secret for verification."""
        return self.fernet.decrypt(encrypted_secret.encode()).decode()

    def get_totp_uri(self, email: str, secret: str) -> str:
        """Generate a provisioning URI for QR codes."""
        return pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name="Manifold")

    def verify_mfa_code(self, secret: str, code: str) -> bool:
        """Verify a TOTP code with clock skew support."""
        if not secret or not code:
            return False
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)

# Global instance for easy access
mfa_service = MFAService()
