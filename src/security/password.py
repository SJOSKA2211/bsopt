"""
Password Security Module
========================

Secure password handling with:
- Bcrypt hashing with configurable rounds
- Password strength validation
- Secure password generation
- Password history tracking
"""

import logging
import re
import secrets
import string
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import pwnedpasswords
from passlib.context import CryptContext

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PasswordValidationResult:
    """Result of password validation."""

    is_valid: bool
    errors: List[str]
    strength_score: int  # 0-100
    suggestions: List[str]


class PasswordValidator:
    """
    Validator for password strength and complexity.
    Checks against common patterns, length, and pwned status.
    """

    def __init__(
        self,
        min_length: Optional[int] = None,
        require_uppercase: Optional[bool] = None,
        require_lowercase: Optional[bool] = None,
        require_digit: Optional[bool] = None,
        require_special: Optional[bool] = None,
    ):
        self.min_length = min_length or settings.PASSWORD_MIN_LENGTH
        self.require_uppercase = (
            require_uppercase
            if require_uppercase is not None
            else settings.PASSWORD_REQUIRE_UPPERCASE
        )
        self.require_lowercase = (
            require_lowercase
            if require_lowercase is not None
            else settings.PASSWORD_REQUIRE_LOWERCASE
        )
        self.require_digit = (
            require_digit if require_digit is not None else settings.PASSWORD_REQUIRE_DIGIT
        )
        self.require_special = (
            require_special if require_special is not None else settings.PASSWORD_REQUIRE_SPECIAL
        )

    def validate(self, password: str, email: Optional[str] = None) -> PasswordValidationResult:
        """
        Validate password strength.

        Args:
            password: Password to validate
            email: Optional email to check for in password

        Returns:
            PasswordValidationResult with validation details
        """
        errors = []
        suggestions = []
        strength_score = 0

        # 1. Length Check
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
            suggestions.append("Use a longer password")
        else:
            strength_score += 20

        # 2. Complexity Checks
        if self.require_uppercase:
            if not re.search(r"[A-Z]", password):
                errors.append("Password must contain at least one uppercase letter")
                suggestions.append("Add uppercase letters")
            else:
                strength_score += 15

        if self.require_lowercase:
            if not re.search(r"[a-z]", password):
                errors.append("Password must contain at least one lowercase letter")
                suggestions.append("Add lowercase letters")
            else:
                strength_score += 15

        if self.require_digit:
            if not re.search(r"\d", password):
                errors.append("Password must contain at least one digit")
                suggestions.append("Add numbers")
            else:
                strength_score += 15

        if self.require_special:
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                errors.append("Password must contain at least one special character")
                suggestions.append("Add special characters (e.g., !, @, #)")
            else:
                strength_score += 15

        # 3. Pattern Checks (Anti-patterns)
        if re.search(r"(.)\1{2,}", password):
            suggestions.append("Avoid repeating the same character 3 or more times")
            strength_score -= 10

        if email and email.split("@")[0].lower() in password.lower():
            errors.append("Password should not contain your email prefix")
            strength_score -= 20

        # 4. Check for pwned password
        try:
            pwned_count = pwnedpasswords.check(password)
            if pwned_count > 0:
                errors.append(
                    f"This password has appeared in a data breach "
                    f"{pwned_count} times and must not be used."
                )
                strength_score = 0  # Major penalty
        except Exception as e:
            logger.warning(f"Could not check pwned status: {e}")

        # Calculate final score
        strength_score = min(100, max(0, strength_score))

        return PasswordValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            strength_score=strength_score,
            suggestions=suggestions,
        )


class PasswordService:
    """
    Secure password hashing and verification.

    Uses bcrypt with configurable rounds for security.
    Supports password history to prevent reuse.
    """

    def __init__(self, rounds: Optional[int] = None):
        # Use bcrypt as specified by settings.BCRYPT_ROUNDS
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=rounds or settings.BCRYPT_ROUNDS,
        )
        self.validator = PasswordValidator()

    def hash_password(self, password: str) -> str:
        """
        Hash a password securely.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        # Truncate to 72 bytes for bcrypt compatibility
        return cast(str, self.pwd_context.hash(password.encode('utf-8')[:72]))

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password to verify
            hashed_password: Stored hash to compare against

        Returns:
            True if password matches, False otherwise
        """
        try:
            # Truncate to 72 bytes for bcrypt compatibility
            return bool(self.pwd_context.verify(plain_password.encode('utf-8')[:72], hashed_password))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def needs_rehash(self, hashed_password: str) -> bool:
        """
        Check if password hash needs to be updated.

        This happens when bcrypt rounds are increased.

        Args:
            hashed_password: Existing password hash

        Returns:
            True if hash should be updated
        """
        return bool(self.pwd_context.needs_update(hashed_password))

    def validate_password(
        self, password: str, email: Optional[str] = None
    ) -> PasswordValidationResult:
        """
        Validate password strength.

        Args:
            password: Password to validate
            email: Optional email for additional checks

        Returns:
            PasswordValidationResult
        """
        return self.validator.validate(password, email)

    def check_password_history(
        self, password: str, password_history: List[str], history_count: int = 5
    ) -> Tuple[bool, str]:
        """
        Check if password was recently used.

        Args:
            password: New password to check
            password_history: List of previous password hashes
            history_count: Number of previous passwords to check

        Returns:
            Tuple of (is_allowed, error_message)
        """
        recent_passwords = password_history[-history_count:]

        for old_hash in recent_passwords:
            if self.verify_password(password, old_hash):
                return False, f"Password was used in the last {history_count} passwords"

        return True, ""

    @staticmethod
    def generate_password(
        length: int = 16,
        include_uppercase: bool = True,
        include_lowercase: bool = True,
        include_digits: bool = True,
        include_special: bool = True,
    ) -> str:
        """
        Generate a secure random password.

        Args:
            length: Password length
            include_uppercase: Include uppercase letters
            include_lowercase: Include lowercase letters
            include_digits: Include numbers
            include_special: Include special characters

        Returns:
            Randomly generated password
        """
        characters = ""
        required = []

        if include_uppercase:
            characters += string.ascii_uppercase
            required.append(secrets.choice(string.ascii_uppercase))
        if include_lowercase:
            characters += string.ascii_lowercase
            required.append(secrets.choice(string.ascii_lowercase))
        if include_digits:
            characters += string.digits
            required.append(secrets.choice(string.digits))
        if include_special:
            special_chars = '!@#$%^&*(),.?":{}|<>'
            characters += special_chars
            required.append(secrets.choice(special_chars))

        if not characters:
            characters = string.ascii_letters + string.digits

        # Generate remaining characters
        remaining_length = length - len(required)
        password_chars = required + [secrets.choice(characters) for _ in range(remaining_length)]

        # Shuffle to avoid predictable positions
        secrets.SystemRandom().shuffle(password_chars)

        return "".join(password_chars)

    @staticmethod
    def generate_reset_token() -> str:
        """
        Generate a secure password reset token.

        Returns:
            URL-safe reset token
        """
        return secrets.token_urlsafe(32)

    @staticmethod
    def generate_verification_token() -> str:
        """
        Generate a secure email verification token.

        Returns:
            URL-safe verification token
        """
        return secrets.token_urlsafe(32)


# Global instance
_password_service_instance: Optional[PasswordService] = None

def get_password_service() -> PasswordService:
    global _password_service_instance
    if _password_service_instance is None:
        # Pass settings.BCRYPT_ROUNDS explicitly
        _password_service_instance = PasswordService(rounds=settings.BCRYPT_ROUNDS)
    return _password_service_instance

# For backward compatibility or direct usage
password_service = get_password_service()