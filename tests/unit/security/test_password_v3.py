from unittest.mock import patch

from src.security.password import PasswordService, PasswordValidator


def test_validator_length():
    validator = PasswordValidator(min_length=10)
    res = validator.validate("short")
    assert not res.is_valid
    assert any("at least 10 characters" in e for e in res.errors)


def test_validator_complexity():
    validator = PasswordValidator(
        require_uppercase=True,
        require_digit=True,
        require_special=True,
        require_lowercase=True,
    )
    res = validator.validate("lowercase")
    assert not res.is_valid
    assert any("uppercase" in e for e in res.errors)

    res = validator.validate("UPPERCASE1")
    assert any("special" in e for e in res.errors)
    assert any("lowercase" in e for e in res.errors)

    # Truly valid and UNIQUE to avoid pwned check failures
    # Mocking pwned check for this specific test case to be safe
    with patch("src.security.password.pwnedpasswords.check", return_value=0):
        res = validator.validate("RickC137_Dimensional_Portal_Gun_2026!")
        assert res.is_valid


def test_validator_email_similarity():
    validator = PasswordValidator()
    res = validator.validate("engineer@bsopt.com", email="engineer@bsopt.com")
    assert not res.is_valid
    assert any("email" in e.lower() for e in res.errors)


@patch("src.security.password.pwnedpasswords.check")
def test_validator_pwned(mock_check):
    mock_check.return_value = 1000  # leaked 1000 times
    validator = PasswordValidator()
    res = validator.validate("password123")
    assert not res.is_valid
    assert any("data breach" in e.lower() for e in res.errors)


def test_password_service_hash_verify():
    service = PasswordService(rounds=4)
    password = "RickC137_Dimensional_Portal_Gun_2026!"
    hashed = service.hash_password(password)
    assert service.verify_password(password, hashed)
    assert not service.verify_password("wrong", hashed)


def test_password_service_generate():
    service = PasswordService(rounds=4)
    pw = service.generate_password(length=16)
    assert len(pw) == 16
    # It should pass service's own validation (which might mock pwned or use a different validator)
    with patch("src.security.password.pwnedpasswords.check", return_value=0):
        res = service.validate_password(pw)
        assert res.is_valid
