import html
import re

def sanitize_string(text: str) -> str:
    """
    Sanitize a string input by escaping HTML characters and trimming whitespace.
    """
    if not text:
        return text
    
    # Escape HTML special characters
    clean_text = html.escape(text)
    
    # Trim whitespace
    clean_text = clean_text.strip()
    
    return clean_text

def sanitize_email(email: str) -> str:
    """
    Normalize email address.
    """
    if not email:
        return email
    return email.lower().strip()

def mask_email(email: str) -> str:
    """
    Mask an email address for logging.
    Example: j***h@example.com
    """
    if not email or "@" not in email:
        return email
    
    try:
        name, domain = email.split("@")
        if len(name) <= 2:
            return f"{name[0]}***@{domain}"
        
        return f"{name[0]}***{name[-1]}@{domain}"
    except Exception:
        return "****@****"