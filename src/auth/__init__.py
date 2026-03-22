"""
Security Module (EquaFlow Phase 2)

Comprehensive security implementation for the BSOPT platform:
- Unified AuthService (Argon2id, TOTP MFA, Asymmetric JWT)
- Role-based access control
- Security utilities and validators
- Audit logging
"""

from src.database.models import AuditLog

from .audit import AuditEvent, log_audit
from .auth import (
    AuthService,
    RoleChecker,
    TokenData,
    TokenPair,
    auth_service,
    get_auth_service,
    get_current_active_user,
    get_current_user,
)

__all__ = [
    # Unified Auth
    "AuthService",
    "auth_service",
    "get_auth_service",
    "get_current_user",
    "get_current_active_user",
    "RoleChecker",
    "TokenData",
    "TokenPair",
    # Audit
    "AuditEvent",
    "log_audit",
    "AuditLog",
]
