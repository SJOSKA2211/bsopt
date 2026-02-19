from src.security.password import PasswordService, PasswordValidator
from tests.test_utils import assert_equal


def test_password_validator():
    validator = PasswordValidator(
        min_length=8, require_uppercase=True, require_digit=True
    )

    # Valid password - using something very unique to avoid pwned check
    result = validator.validate("XyZ_987_!!_Unique_2025")
    assert result.is_valid

    # Invalid: too short
    result = validator.validate("Sh1!")
    assert not result.is_valid
    assert any("at least 8 characters" in e for e in result.errors)

    # Invalid: missing uppercase
    result = validator.validate("weakpass123")
    assert not result.is_valid
    assert any("uppercase letter" in e for e in result.errors)


def test_password_service_hashing():
    service = PasswordService(rounds=4)  # Use few rounds for speed in tests
    password = "MySecurePassword123!"

    hashed = service.hash_password(password)
    assert hashed != password
    assert service.verify_password(password, hashed)
    assert not service.verify_password("wrong password", hashed)


def test_password_generation():
    password = PasswordService.generate_password(length=20)
    assert_equal(len(password), 20)
    # Check if it has mixed characters (statistically likely)
    assert any(c.isupper() for c in password)
    assert any(c.islower() for c in password)
    assert any(c.isdigit() for c in password)


def test_password_history():
    service = PasswordService(rounds=4)
    old_passwords = [service.hash_password(f"Pass{i}!") for i in range(3)]

    # Test recent password
    is_allowed, msg = service.check_password_history("Pass1!", old_passwords)
    assert not is_allowed
    assert "used in the last" in msg

    # Test new password
    is_allowed, msg = service.check_password_history("NewPass123!", old_passwords)
    assert is_allowed
    assert_equal(msg, "")


def test_password_long_password():
    """Test handling of passwords longer than bcrypt's 72-byte limit."""
    service = PasswordService(rounds=4)
    long_password = "a" * 100

    # Argon2 handles long passwords correctly without truncation
    hashed = service.hash_password(long_password)
    assert service.verify_password(long_password, hashed)

    # Unlike bcrypt, Argon2 should NOT match if only the first 72 chars are the same
    same_start = ("a" * 72) + "different"
    assert not service.verify_password(same_start, hashed)

    different_start = ("b" * 72) + "same_end"
    assert not service.verify_password(different_start, hashed)
