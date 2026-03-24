"""
API Routes

Modular API route definitions:
- Authentication routes
- User management routes
- Pricing routes
- Admin routes
- Debug routes # Added for debug_router
"""

from services.api.routes.auth import router as auth_router
from services.api.routes.debug import router as debug_router
from services.api.routes.market import router as market_router
from services.api.routes.ml import router as ml_router
from services.api.routes.options import router as options_router
from services.api.routes.portfolio import router as portfolio_router
from services.api.routes.pricing import router as pricing_router
from services.api.routes.system import router as system_router
from services.api.routes.users import router as users_router
from services.api.routes.websocket import router as websocket_router

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
    "market_router",
]
