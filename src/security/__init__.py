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
from .password import PasswordService, PasswordValidator

__all__ = [
    # Password
    "PasswordService",
    "PasswordValidator",
    # Audit
    "AuditEvent",
    "log_audit",
    "AuditLog",
]
