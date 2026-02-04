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

import bcrypt
import pwnedpasswords

from src.config import settings

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

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
        # Handle cases where settings might be None (e.g. during test initialization)
        s = settings if settings else None
        
        self.min_length = min_length or getattr(s, "PASSWORD_MIN_LENGTH", 8)
        self.require_uppercase = (
            require_uppercase
            if require_uppercase is not None
            else getattr(s, "PASSWORD_REQUIRE_UPPERCASE", True)
        )
        self.require_lowercase = (
            require_lowercase
            if require_lowercase is not None
            else getattr(s, "PASSWORD_REQUIRE_LOWERCASE", True)
        )
        self.require_digit = (
            require_digit if require_digit is not None else getattr(s, "PASSWORD_REQUIRE_DIGIT", True)
        )
        self.require_special = (
            require_special if require_special is not None else getattr(s, "PASSWORD_REQUIRE_SPECIAL", False)
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

    Uses Argon2id for new hashes, with fallback to bcrypt for verification.
    """

    def __init__(self, rounds: Optional[int] = None):
        # Default to settings if not provided
        self.rounds = rounds or (settings.BCRYPT_ROUNDS if settings else 12)
        self.validator = PasswordValidator()
        
        # Initialize with tuned parameters for Argon2id
        # Use explicit type checking to handle cases where settings might be mocked (e.g. in tests)
        argon2_time_cost = getattr(settings, "ARGON2_TIME_COST", 3)
        if not isinstance(argon2_time_cost, int):
            argon2_time_cost = 3
            
        argon2_memory_cost = getattr(settings, "ARGON2_MEMORY_COST", 65536)
        if not isinstance(argon2_memory_cost, int):
            argon2_memory_cost = 65536
            
        argon2_parallelism = getattr(settings, "ARGON2_PARALLELISM", 4)
        if not isinstance(argon2_parallelism, int):
            argon2_parallelism = 4

        self.ph = PasswordHasher(
            time_cost=argon2_time_cost,
            memory_cost=argon2_memory_cost,
            parallelism=argon2_parallelism
        )

    def hash_password(self, password: str) -> str:
        """
        Hash a password securely using Argon2id.
        """
        return self.ph.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash (supports Argon2 and Bcrypt).
        """
        if not plain_password or not hashed_password:
            return False
            
        # 1. Try Argon2 (new standard)
        if hashed_password.startswith("$argon2"):
            try:
                return self.ph.verify(hashed_password, plain_password)
            except VerifyMismatchError:
                return False
            except Exception as e:
                logger.error(f"Argon2 verification error: {e}")
                return False
        
        # 2. Try Bcrypt (fallback for legacy hashes)
        try:
            if isinstance(plain_password, str):
                plain_password = plain_password.encode("utf-8")
            if isinstance(hashed_password, str):
                hashed_password = hashed_password.encode("utf-8")
                
            # Bcrypt has a 72-byte limit.
            if len(plain_password) > 72:
                plain_password = plain_password[:72]
                
            return bcrypt.checkpw(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Legacy Bcrypt verification error: {e}")
            return False

    def needs_rehash(self, hashed_password: str) -> bool:
        """
        Check if password hash needs to be updated to Argon2.
        """
        if not hashed_password:
            return True
        
        # Check if it's a legacy bcrypt hash
        if not hashed_password.startswith("$argon2"):
            return True
            
        # Check if Argon2 parameters need tuning
        try:
            return self.ph.check_needs_rehash(hashed_password)
        except Exception:
            return True

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
        # Handle cases where settings might be None (e.g. during early test initialization)
        try:
            rounds = settings.BCRYPT_ROUNDS if settings else 12
        except AttributeError:
            rounds = 12
        _password_service_instance = PasswordService(rounds=rounds)
    return _password_service_instance

# For backward compatibility or direct usage
password_service = get_password_service()
