import pytest
from src.utils.sanitization import sanitize_string, sanitize_email, mask_email

def test_sanitize_string():
    assert sanitize_string("  <script>alert(1)</script>  ") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert sanitize_string(None) is None
    assert sanitize_string("") == ""

def test_sanitize_email():
    assert sanitize_email("  Test@Example.com  ") == "test@example.com"
    assert sanitize_email(None) is None

def test_mask_email():
    assert mask_email("joseph@example.com") == "j***h@example.com"
    assert mask_email("ab@example.com") == "a***@example.com"
    assert mask_email("a@example.com") == "a***@example.com"
    assert mask_email("invalid-email") == "invalid-email"
