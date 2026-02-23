import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import ORJSONResponse
from strawberry.fastapi import GraphQLRouter

from src.api.websockets.manager import manager as ws_manager
from src.shared.observability import logging_middleware, setup_logging, tune_gc
from src.streaming.graphql.schema import schema

# Optimized event loop
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize components
    setup_logging()
    tune_gc()

    # Initialize Redis
    from src.config import settings
    from src.utils.cache import init_redis_cache

    await init_redis_cache(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
    )

    yield


app = FastAPI(
    title="BS-Opt Market Data Service",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)
app.middleware("http")(logging_middleware)

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


@app.get("/health")
async def health():
    return {"status": "healthy"}





@app.websocket("/marketdata/{symbol}")
async def websocket_marketdata(websocket: WebSocket, symbol: str):
    """
    High-speed market data WebSocket endpoint.
    OPTIMIZED: Uses the central ConnectionManager and Redis-backed Pub/Sub.
    """
    try:
        # Accept and subscribe to the requested symbol
        await ws_manager.connect(websocket, symbol.upper())

        while True:
            # Keep connection alive
            await websocket.receive_text()

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, symbol.upper())
    except Exception as e:
        logger.error("ws_error", symbol=symbol, error=str(e))
        ws_manager.disconnect(websocket, symbol.upper())
