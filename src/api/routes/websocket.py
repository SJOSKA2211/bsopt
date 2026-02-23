import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.api.websockets.manager import ProtocolType, manager

# In a real app, we would import auth dependencies here
# from src.api.dependencies import get_current_user

logger = structlog.get_logger()
router = APIRouter()


@router.websocket("/ws/market-data")
async def market_data_ws(
    websocket: WebSocket,
    symbol: str = Query(..., description="Ticker symbol to subscribe to"),
    protocol: ProtocolType = Query(ProtocolType.JSON),
):
    """
    WebSocket endpoint for real-time market data.
    OPTIMIZED: Metadata-first connection to prevent protocol race conditions.
    """
    # 1. Initialize Metadata FIRST
    from src.api.websockets.manager import ConnectionMetadata

    websocket.metadata = ConnectionMetadata(protocol=protocol)

    # 2. Connect to symbol-aware manager
    await manager.connect(websocket, symbol.upper())

    try:
        while True:
            # 3. Robust Keep-Alive (Supports both Text and Binary frames)
            # We don't process incoming commands yet, just waiting for disconnect
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("ws_route_error", symbol=symbol, error=str(e))
    finally:
        manager.disconnect(websocket, symbol.upper())
