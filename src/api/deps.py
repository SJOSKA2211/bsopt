"""
API dependency re-exports.

This module bridges older imports like ``src.api.deps.get_current_user`` to the
centralized authentication helpers in ``src.security.auth`` so that route
modules do not need to know where the implementations live.
"""

from src.security.auth import (  # noqa: F401
    get_current_active_user,
    get_current_user,
    get_current_user_flexible,
    require_tier,
)

__all__ = [
    "get_current_user",
    "get_current_active_user",
    "get_current_user_flexible",
    "require_tier",
]

