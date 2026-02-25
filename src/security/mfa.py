"""
MFA Service
===========

Handles Multi-Factor Authentication using Time-based One-Time Passwords (TOTP).
"""

import io
from base64 import b64encode

import pyotp
import qrcode


class MfaService:
    def __init__(self, issuer_name: str = "BSOPT Platform"):
        self.issuer_name = issuer_name

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
        """Verify a TOTP code against the secret."""
        totp = pyotp.TOTP(secret)
        return totp.verify(code)


# Global MFA service instance
mfa_service = MfaService()
