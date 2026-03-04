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
    token: str = Query(None, description="Bearer token for authentication"),
):
    """
    WebSocket endpoint for real-time market data.
    OPTIMIZED: Metadata-first connection to prevent protocol race conditions.
    """
    # Authenticate the WebSocket connection
    if not token:
        await websocket.close(code=1008, reason="Authentication required")
        return

    try:
        from src.security.auth import auth_service

        await auth_service.validate_token(token)
    except Exception:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return

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
