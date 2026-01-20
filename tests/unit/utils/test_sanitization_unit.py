import pytest
from src.utils.sanitization import sanitize_string, sanitize_email, mask_email

def test_sanitize_string():
    # html.escape in Python 3.13 escapes single quotes as &#x27;
    assert sanitize_string("<script>alert('xss')</script>") == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
    assert sanitize_string("  hello  ") == "hello"
    assert sanitize_string(None) is None
    assert sanitize_string("") == ""

def test_sanitize_email():
    assert sanitize_email("  User@Example.com  ") == "user@example.com"
    assert sanitize_email("") == ""
    assert sanitize_email(None) is None

def test_mask_email():
    assert mask_email("john.doe@example.com") == "j***e@example.com"
    assert mask_email("ab@example.com") == "a***@example.com"
    assert mask_email("a@example.com") == "a***@example.com"
    assert mask_email("invalid-email") == "invalid-email"
    assert mask_email(None) is None
    assert mask_email("") == ""

def test_mask_email_exception():
    # Pass something that has '@' but is not a string to trigger Exception
    class BadString:
        def __contains__(self, item): return item == "@"
        def split(self, item): raise Exception("Manual fail")
        def __str__(self): return "bad@string"
        
    assert mask_email(BadString()) == "****@****"