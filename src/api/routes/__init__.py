"""
API Routes

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
from .options import router as options_router
from .portfolio import router as portfolio_router
from .pricing import router as pricing_router
from .system import router as system_router
from .users import router as users_router
from .websocket import router as websocket_router

# Added missing routers for consistency

__all__ = [
    "auth_router",
    "ml_router",
    "users_router",
    "pricing_router",
    "options_router",
    "portfolio_router",
    "debug_router",
    "system_router",
    "websocket_router",
]
