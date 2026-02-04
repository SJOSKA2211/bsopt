from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import ORJSONResponse
from strawberry.fastapi import GraphQLRouter
from src.streaming.graphql.schema import schema
from src.shared.observability import logging_middleware, setup_logging, tune_gc
from contextlib import asynccontextmanager
import asyncio
import orjson
import structlog

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
    from src.utils.cache import init_redis_cache
    from src.config import settings
    await init_redis_cache(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD
    )
    
    yield

app = FastAPI(
    title="BS-Opt Market Data Service",
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)
app.middleware("http")(logging_middleware)

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.websocket("/marketdata")
async def websocket_marketdata(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send mock market data
            mock_data = {
                "symbol": "AAPL",
                "time": int(asyncio.time()),
                "open": 150 + (asyncio.time() % 10),
                "high": 160 + (asyncio.time() % 10),
                "low": 140 + (asyncio.time() % 10),
                "close": 150 + (asyncio.time() % 10)
            }
            await websocket.send_text(orjson.dumps(mock_data).decode('utf-8'))
            await asyncio.sleep(1)  # Send data every second
    except WebSocketDisconnect:
        logger.info("ws_client_disconnected", channel="/marketdata")
    except Exception as e:
        logger.error("ws_error", channel="/marketdata", error=str(e))