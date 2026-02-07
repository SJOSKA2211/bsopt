import sys
from unittest.mock import MagicMock

import pytest

# Mock qrcode if not available
try:
    import qrcode
except ImportError:
    sys.modules["qrcode"] = MagicMock()

from src.security.mfa import MfaService


@pytest.fixture
def mfa_service():
    return MfaService()


def test_mfa_secret_generation(mfa_service):
    secret = mfa_service.generate_secret()
    assert len(secret) == 32
    assert isinstance(secret, str)


def test_mfa_provisioning_uri(mfa_service):
    secret = mfa_service.generate_secret()
    uri = mfa_service.get_provisioning_uri("test@example.com", secret)
    assert "otpauth://totp/" in uri
    # Email might be encoded as test%40example.com
    assert "test%40example.com" in uri or "test@example.com" in uri


def test_mfa_verification(mfa_service):
    import pyotp

    secret = mfa_service.generate_secret()
    totp = pyotp.totp.TOTP(secret)
    code = totp.now()

    assert mfa_service.verify_code(secret, code) is True
    assert mfa_service.verify_code(secret, "000000") is False
