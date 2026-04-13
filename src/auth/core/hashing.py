"""
Secure Hashing Substrate (Argon2id).
"""

import logging

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from src.shared.config import settings

logger = logging.getLogger(__name__)


class PasswordHasherService:
    """
    Argon2id password hashing and verification.
    """

    def __init__(self):
        self.ph = PasswordHasher(
            time_cost=settings.ARGON2_TIME_COST,
            memory_cost=settings.ARGON2_MEMORY_COST,
            parallelism=settings.ARGON2_PARALLELISM,
        )
        self.DUMMY_HASH = self.ph.hash("static-dummy-password-for-timing-protection")

    def hash_password(self, password: str) -> str:
        """Hash a password using Argon2id."""
        return self.ph.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against an Argon2id hash."""
        try:
            return self.ph.verify(hashed_password, plain_password)
        except VerifyMismatchError:
            return False
        except Exception as e:
            logger.error(f"password_verification_error: {e}")
            return False

    def needs_rehash(self, hashed_password: str) -> bool:
        """Check if a hash needs to be updated to current Argon2id parameters."""
        if not hashed_password.startswith("$argon2"):
            return True
        try:
            return self.ph.check_needs_rehash(hashed_password)
        except Exception:
            return True


# Global instance for easy access
hasher = PasswordHasherService()