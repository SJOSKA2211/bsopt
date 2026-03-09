import asyncio
import os
from contextlib import asynccontextmanager

import structlog
import uvloop
from brotli_asgi import BrotliMiddleware
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from src.api.graphql.schema import schema
from src.api.middleware.security import (
    CSRFMiddleware,
    InputSanitizationMiddleware,
    IPBlockMiddleware,
    JWTAuthenticationMiddleware,
    SecurityHeadersMiddleware,
)
from src.api.responses import MsgspecJSONResponse
from src.api.routes import (
    auth_router,
    debug_router,
    ml_router,
    options_router,
    portfolio_router,
    pricing_router,
    users_router,
    websocket_router,
)
from src.config import settings
from src.security.auth import RoleChecker
from src.shared.observability import logging_middleware, start_system_metrics_loop

# Initialize logging
logger = structlog.get_logger()


def get_context(request: Request) -> dict:
    """GraphQL Context factory."""
    if os.getenv("TESTING") == "true":
        return {}
    return {"request": request}


# Optimized event loop
try:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except (ImportError, AttributeError):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    God-Mode Lifespan: Handles resource lifecycle with zero-leak guarantees.
    """
    start_system_metrics_loop("api")

    # Initialize Database (Weaponizer God-Mode)
    from src.database import db_manager

    db_manager.initialize()

    # Initialize Redis
    from src.utils.cache import init_redis_cache

    await init_redis_cache()

    # Chaos Injection
    from src.utils.chaos import monkey

    if monkey.enabled:
        logger.warning("chaos_mode_active_injecting_startup_latency")
        await monkey.delay_db(0.5)

    yield

    # Shutdown
    from src.api.websockets.manager import manager
    from src.database import dispose_engine
    from src.utils.cache import close_redis_cache

    await manager.close()
    await dispose_engine()
    await close_redis_cache()
    logger.info("api_shutdown_complete_database_engines_disposed")


app = FastAPI(
    title=settings.PROJECT_NAME,
    default_response_class=MsgspecJSONResponse,
    lifespan=lifespan,
)


# Middleware
app.add_middleware(BrotliMiddleware, minimum_size=1000, quality=4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-CSRF-Token",
        "X-Request-ID",
        "X-API-Key",
        "Accept",
    ],
)
app.middleware("http")(logging_middleware)


# Exception Handler
async def api_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    from src.api.exceptions import BaseAPIException

    if isinstance(exc, BaseAPIException):
        return MsgspecJSONResponse(
            status_code=getattr(exc, "status_code", 500),
            content={
                "error": getattr(exc, "error_code", getattr(exc, "error", "InternalServerError")),
                "message": getattr(exc, "message", str(exc)),
                "details": getattr(exc, "details", None),
            },
        )

    if isinstance(exc, HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        return MsgspecJSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "status_code": exc.status_code,
                **detail,
            },
            headers=getattr(exc, "headers", None),
        )

    error_detail = str(exc)
    if settings.ENVIRONMENT != "prod":
        import traceback

        error_detail = traceback.format_exc()
        logger.error("api_error_detailed", error=error_detail, path=request.url.path)
    else:
        logger.error("api_error", error=str(exc), path=request.url.path)

    return MsgspecJSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": (
                error_detail if settings.ENVIRONMENT != "prod" else "An unexpected error occurred"
            ),
        },
    )


from fastapi.exceptions import RequestValidationError  # noqa: E402


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI built-in validation errors."""
    return MsgspecJSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


app.add_exception_handler(Exception, api_exception_handler)
app.add_exception_handler(HTTPException, api_exception_handler)


graphql_app = GraphQLRouter(schema)

# Security Middleware (Order matters: executed from bottom to top of this list)
# 1. JWT Auth (innermost - specific to routes)
app.add_middleware(JWTAuthenticationMiddleware)
# 2. CSRF Protection
app.add_middleware(CSRFMiddleware)
# 3. Input Sanitization (logging only)
app.add_middleware(InputSanitizationMiddleware)
# 4. IP Blocking (reject bad IPs early)
app.add_middleware(IPBlockMiddleware)
# 5. Security Headers (outermost - applies to all responses)
app.add_middleware(SecurityHeadersMiddleware)

# API v1 Routes
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth_router)
api_router.include_router(pricing_router)
api_router.include_router(ml_router)
api_router.include_router(options_router)
api_router.include_router(portfolio_router)
api_router.include_router(users_router)
api_router.include_router(websocket_router)  # Include websocket router

if settings.ENVIRONMENT not in ("prod", "production"):
    api_router.include_router(debug_router)

app.include_router(api_router)
app.include_router(graphql_app, prefix="/graphql")


@app.get("/health")
@app.get("/api/v1/health")
async def health():
    from src.database import health_check

    return {"status": "healthy", "database": health_check()}


@app.get("/api/diagnostics/imports")
async def diagnostics_imports():
    return {"successful_imports": True}


@app.get("/admin-only")
async def admin_only(user: dict = Depends(RoleChecker(["admin"]))):
    return {"message": "Welcome, Admin"}


@app.get("/metrics")
async def metrics(request: Request):
    # Only allow internal/authenticated access to metrics
    if settings.is_production:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            from src.api.responses import MsgspecJSONResponse

            return MsgspecJSONResponse(status_code=401, content={"detail": "Not authenticated"})
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {"message": "BS-Opt Optimized API is running"}
