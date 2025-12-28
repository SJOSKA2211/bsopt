import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import auth_router, pricing_router, users_router
from src.config import settings
from src.pricing.black_scholes import BlackScholesEngine
from src.security.auth import token_blacklist
from src.utils.cache import (
    close_redis_cache,
    init_redis_cache,
    publish_to_redis,
    redis_channel_updates,
)

# Global state
active_connections: Dict[str, WebSocket] = {}
redis_client: Any = None
redis_pubsub: Any = None

logger = logging.getLogger(__name__)

app = FastAPI(title="BSOPT API", version="2.1.0")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    # Initialize Redis
    redis_client = await init_redis_cache(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
    )
    if redis_client:
        await token_blacklist.initialize(redis_client)
        logger.info("Token blacklist initialized with Redis.")
    else:
        logger.warning("Redis not available. Token blacklist is in-memory.")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    await close_redis_cache()
    logger.info("Redis cache connection closed.")


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
app.include_router(pricing_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc), "version": "2.1.0"}
