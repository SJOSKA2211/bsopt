import tracemalloc # Added for tracemalloc
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

tracemalloc.start(1024) # Added for tracemalloc with 1024 frames

from fastapi import FastAPI, HTTPException, Request, WebSocket, status, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routes import auth_router, ml_router, pricing_router, users_router, debug_router # Added debug_router
from src.api.schemas.common import ErrorDetail, ErrorResponse, HealthResponse
from src.config import settings
from src.pricing.black_scholes import BlackScholesEngine
from src.security.auth import token_blacklist
from src.utils.cache import (
    close_redis_cache,
    init_redis_cache,
    publish_to_redis,
    redis_channel_updates,
)
from src.tasks.graceful_shutdown import shutdown_manager

# Global state
active_connections: Dict[str, WebSocket] = {}
redis_client: Any = None
redis_pubsub: Any = None

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BSOPT API",
    version="2.2.0",
    description="Advanced Black-Scholes Option Pricing API",
)

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

from src.api.exceptions import BaseAPIException, InternalServerException, ValidationException

# --- Exception Handlers ---


@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(
            ErrorResponse(
                error=exc.error_code,
                message=exc.message,
                details=exc.details,
                request_id=request_id,
            )
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(
            ErrorResponse(
                error="HTTPError",
                message=str(exc.detail),
                request_id=request_id,
            )
        ),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", None)
    details = [
        ErrorDetail(
            field=" -> ".join(map(str, error["loc"])),
            message=error["msg"],
            code=error["type"],
        )
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(
            ErrorResponse(
                error="ValidationError",
                message="Request validation failed",
                details=details,
                request_id=request_id,
            )
        ),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    request_id = getattr(request.state, "request_id", None)
    logger.exception(f"Unhandled exception [RID: {request_id}]: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(
            ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                request_id=request_id,
            )
        ),
    )


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign a unique ID to every request."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their outcomes."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        f"RID: {request_id} Method: {request.method} Path: {request.url.path} "
        f"Status: {response.status_code} Duration: {process_time:.2f}ms"
    )
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add standard security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    # Install signal handlers for graceful shutdown
    shutdown_manager.install_signal_handlers()
    shutdown_manager.register_cleanup(close_redis_cache)
    
    # Initialize Redis
    redis_client = await init_redis_cache(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
    )
    app.state.redis_client = redis_client # Store redis_client on app.state

    if redis_client:
        await token_blacklist.initialize(redis_client)
        logger.info("Token blacklist initialized with Redis.")
        
        # Start Redis Pub/Sub listener
        redis_pubsub_task = asyncio.create_task(redis_pubsub_listener())
        shutdown_manager.register_cleanup(redis_pubsub_task.cancel)
        logger.info("Redis pub/sub listener started.")

        # Start simulated pricing publisher
        simulated_pricing_task = asyncio.create_task(simulated_pricing_publisher())
        shutdown_manager.register_cleanup(simulated_pricing_task.cancel)
        logger.info("Simulated pricing publisher started.")

    else:
        logger.warning("Redis not available. Token blacklist is in-memory, and Redis-dependent tasks will not start.")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    await close_redis_cache()
    logger.info("Redis cache connection closed.")


# Dependency to get Redis client
async def get_redis_client() -> Any:
    if not app.state.redis_client:
        raise HTTPException(status_code=500, detail="Redis client not initialized")
    return app.state.redis_client

# Function to broadcast messages to all connected clients
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast a JSON message to all connected WebSocket clients."""
    encoded_message = json.dumps(message)

    disconnected_clients = []
    for connection_id, ws in list(active_connections.items()):
        try:
            await ws.send_text(encoded_message)
        except Exception as e:
            logger.warning(f"WebSocket send error to {connection_id}: {e}")
            disconnected_clients.append(connection_id)

    for connection_id in disconnected_clients:
        if connection_id in active_connections:
            del active_connections[connection_id]


# Redis Pub/Sub listener task
async def redis_pubsub_listener():
    """Listens to Redis pub/sub channel and broadcasts messages via WebSockets."""
    if not redis_pubsub:
        logger.error("Redis pub/sub client not initialized.")
        return

    await redis_pubsub.subscribe(redis_channel_updates)
    logger.info(f"Subscribed to Redis channel: {redis_channel_updates}")

    try:
        while True:
            message = await redis_pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("type") == "message":
                try:
                    data = json.loads(message["data"])
                    await broadcast_message(data)
                except Exception as e:
                    logger.error(f"Error processing Redis message: {e}")
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        logger.info("Redis pub/sub listener task cancelled.")
    finally:
        if redis_pubsub:
            await redis_pubsub.unsubscribe(redis_channel_updates)


# Simulated pricing data publisher task
async def simulated_pricing_publisher():
    """Periodically simulates pricing data changes and publishes them to Redis."""
    import numpy as np # Moved import here
    publish_interval_seconds = 5

    while True:
        try:
            instrument_params = {
                "spot": float(np.random.uniform(90.0, 110.0)),
                "strike": 100.0,
                "maturity": float(np.random.uniform(0.1, 1.0)),
                "volatility": float(np.random.uniform(0.1, 0.5)),
                "rate": 0.05,
                "dividend": 0.02,
                "option_type": str(np.random.choice(["call", "put"])),
            }

            start_time = time.perf_counter()
            price = BlackScholesEngine.price_options(**instrument_params)
            greeks = BlackScholesEngine.calculate_greeks(**instrument_params)
            calc_time = (time.perf_counter() - start_time) * 1000

            simulated_price_data = {
                "type": "price_update",
                "payload": {
                    "instrument": {
                        "id": (
                            f"SIM_{instrument_params['option_type']}_"
                            f"{np.random.randint(1000, 9999)}"
                        ),
                        "params": instrument_params,
                    },
                    "price": float(price),
                    "greeks": (
                        greeks
                        if isinstance(greeks, dict)
                        else {
                            "delta": greeks.delta,
                            "gamma": greeks.gamma,
                            "theta": greeks.theta,
                            "vega": greeks.vega,
                            "rho": greeks.rho,
                        }
                    ),
                    "calculated_at": datetime.now(timezone.utc).isoformat(),
                    "computation_time_ms": calc_time,
                },
            }
            await publish_to_redis(redis_channel_updates, simulated_price_data)
            await asyncio.sleep(publish_interval_seconds)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in simulated pricing publisher: {e}")
            await asyncio.sleep(publish_interval_seconds)


# Routes
app.include_router(auth_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(ml_router, prefix="/api/v1")
app.include_router(pricing_router, prefix="/api/v1")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """
    Detailed health check including database and cache status.
    """
    checks = {
        "database": {"status": "healthy"},
        "redis": {"status": "healthy" if redis_client else "unhealthy"},
    }
    
    # Simple DB check
    from src.database import engine
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}

    overall_status = "healthy" if all(c["status"] == "healthy" for c in checks.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="2.2.0",
        checks=checks
    )


@app.get("/health")
async def root_health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc), "version": "2.2.0"}

