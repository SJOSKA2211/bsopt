"""
Security Module
===============

Comprehensive security implementation for the BSOPT platform:
- JWT authentication with access and refresh tokens
- Password hashing with bcrypt
- Role-based access control
- Security utilities and validators
- Audit logging
"""

from .audit import AuditEvent, AuditLog, log_audit
from .auth import (
    AuthService,
    TokenData,
    TokenPair,
    get_current_active_user,
    get_current_user,
    require_tier,
)
from .password import PasswordService, PasswordValidator

__all__ = [
    # Auth
    "AuthService",
    "get_current_user",
    "get_current_active_user",
    "require_tier",
    "TokenData",
    "TokenPair",
    # Password
    "PasswordService",
    "PasswordValidator",
    # Audit
    "AuditEvent",
    "log_audit",
    "AuditLog",
]
