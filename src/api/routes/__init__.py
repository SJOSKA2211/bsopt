"""
API Routes
==========

Modular API route definitions:
- Authentication routes
- User management routes
- Pricing routes
- Admin routes
"""

from .auth import router as auth_router
from .ml import router as ml_router
from .pricing import router as pricing_router
from .users import router as users_router

__all__ = [
    "auth_router",
    "ml_router",
    "users_router",
    "pricing_router",
]
