"""
API Routes
==========

Modular API route definitions:
- Authentication routes
- User management routes
- Pricing routes
- Admin routes
- Debug routes # Added for debug_router
"""

from .auth import router as auth_router
from .debug import router as debug_router
from .ml import router as ml_router
from .pricing import router as pricing_router
from .system import router as system_router
from .users import router as users_router

# Added debug_router

__all__ = [
    "auth_router",
    "ml_router",
    "users_router",
    "pricing_router",
    "debug_router",  # Added debug_router
]
