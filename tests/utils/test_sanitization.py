import pytest
from src.utils.sanitization import sanitize_string, sanitize_email, mask_email

def test_sanitize_string():
    assert sanitize_string("  <script>alert(1)</script>  ") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert sanitize_string("") == ""
    assert sanitize_string(None) is None

def test_sanitize_email():
    assert sanitize_email("  Test@Example.COM  ") == "test@example.com"
    assert sanitize_email("") == ""
    assert sanitize_email(None) is None

def test_mask_email():
    assert mask_email("johndoe@example.com") == "j***e@example.com"
    assert mask_email("ab@example.com") == "a***@example.com"
    assert mask_email("a@example.com") == "a***@example.com"
    assert mask_email("invalid-email") == "invalid-email"
    assert mask_email("") == ""
    assert mask_email(None) is None
    
    # Test split exception handling
    assert mask_email("weird@@@email") == "****@****"