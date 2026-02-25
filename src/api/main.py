import asyncio

import structlog
import uvloop
from brotli_asgi import BrotliMiddleware
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from src.api.graphql.schema import schema
from src.api.middleware.security import (
    InputSanitizationMiddleware,
    JWTAuthenticationMiddleware,
    SecurityHeadersMiddleware,
)
from src.api.routes.auth import router as auth_router
from src.api.routes.debug import router as debug_router
from src.api.routes.ml import router as ml_router
from src.api.routes.options import router as options_router
from src.api.routes.portfolio import router as portfolio_router
from src.api.routes.pricing import router as pricing_router
from src.api.routes.users import router as users_router
from src.config import settings
from src.shared.observability import logging_middleware, start_system_metrics_loop

# Initialize logging
logger = structlog.get_logger()

# Optimized event loop
try:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except (ImportError, AttributeError):
    pass

app = FastAPI(title=settings.PROJECT_NAME, default_response_class=ORJSONResponse)


@app.on_event("startup")
async def startup_event():
    start_system_metrics_loop("api")

    # Chaos Injection
    from src.utils.chaos import monkey

    if monkey.enabled:
        logger.warning("chaos_mode_active_injecting_startup_latency")
        await monkey.delay_db(0.5)  # Slight delay to trigger latency detectors without timeout


# Middleware
app.add_middleware(BrotliMiddleware, minimum_size=1000, quality=4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(logging_middleware)


# Exception Handler
async def api_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    from src.api.exceptions import BaseAPIException

    if isinstance(exc, BaseAPIException):
        return ORJSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    if isinstance(exc, HTTPException):
        return ORJSONResponse(
            status_code=exc.status_code,
            content=(exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}),
            headers=getattr(exc, "headers", None),
        )

    error_detail = str(exc)
    if settings.ENVIRONMENT != "prod":
        import traceback

        error_detail = traceback.format_exc()
        logger.error("api_error_detailed", error=error_detail, path=request.url.path)
    else:
        logger.error("api_error", error=str(exc), path=request.url.path)

    return ORJSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": (
                error_detail if settings.ENVIRONMENT != "prod" else "An unexpected error occurred"
            ),
        },
    )


app.add_exception_handler(Exception, api_exception_handler)
app.add_exception_handler(HTTPException, api_exception_handler)


graphql_app = GraphQLRouter(schema)

# Security Middleware (Order matters: executed from bottom to top of this list)
# 1. JWT Auth (innermost - specific to routes)
app.add_middleware(JWTAuthenticationMiddleware)
# 2. Input Sanitization (logging only)
app.add_middleware(InputSanitizationMiddleware)
# 3. Security Headers (outermost - applies to all responses)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(pricing_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(options_router, prefix="/api/v1")
app.include_router(portfolio_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(debug_router, prefix="/api/v1")
app.include_router(graphql_app, prefix="/graphql")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {"message": "BS-Opt Optimized API"}
