import pytest

from src.security.password import PasswordService, PasswordValidator


@pytest.fixture
def password_service():
    return PasswordService()


def test_hash_and_verify_password(password_service):
    password = "StrongPassword123!"
    hashed = password_service.hash_password(password)
    assert hashed != password
    assert password_service.verify_password(password, hashed) is True
    assert password_service.verify_password("wrong", hashed) is False


def test_password_validator():
    validator = PasswordValidator(min_length=8)

    # Valid - using a likely non-pwned password
    res = validator.validate("Complex_Pass_2025_!$")
    assert res.is_valid is True

    # Too short
    res = validator.validate("Short1!")
    assert res.is_valid is False
    assert any("at least 8" in e for e in res.errors)

    # No digit
    res = validator.validate("NoDigits!")
    assert res.is_valid is False


def test_generate_password():
    pwd = PasswordService.generate_password(length=20)
    assert len(pwd) == 20
    # Should have different types of characters
    assert any(c.isdigit() for c in pwd)
    assert any(c.isupper() for c in pwd)


def test_password_history(password_service):
    old_passwords = [
        password_service.hash_password("Pass123!"),
        password_service.hash_password("Pass456!"),
    ]

    # Reuse old password
    allowed, msg = password_service.check_password_history("Pass123!", old_passwords)
    assert allowed is False
    assert "last" in msg

    # New password
    allowed, msg = password_service.check_password_history("NewPass789!", old_passwords)
    assert allowed is True
