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
    symbol: str | None = Query(None, description="Initial symbol to subscribe to"),
    protocol: ProtocolType = Query(ProtocolType.JSON),
    token: str = Query(None, description="Bearer token for authentication"),
):
    """
    WebSocket endpoint for real-time market data.
    OPTIMIZED: Metadata-first connection with dynamic command handling.
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

    # 1. Initialize Metadata
    from src.api.websockets.manager import ConnectionMetadata
    websocket.metadata = ConnectionMetadata(protocol=protocol)

    # 2. Connect to manager
    await manager.connect(websocket)

    # 3. Initial subscription if provided
    if symbol:
        await manager.subscribe_to_symbol(websocket, symbol)

    try:
        while True:
            # 4. Dynamic Command Handling
            if protocol == ProtocolType.MSGPACK:
                data = await websocket.receive_bytes()
                msg = WebSocketCodec.decode(data, protocol)
            else:
                msg = await websocket.receive_json()

            if not isinstance(msg, dict):
                continue

            action = msg.get("action")
            
            if action == "subscribe":
                sym = msg.get("symbol")
                if sym:
                    await manager.subscribe_to_symbol(websocket, sym)
            
            elif action == "unsubscribe":
                sym = msg.get("symbol")
                if sym:
                    await manager.unsubscribe_from_symbol(websocket, sym)
            
            elif action == "heartbeat":
                websocket.metadata.update_heartbeat()
                # Echo heartbeat back
                if protocol == ProtocolType.MSGPACK:
                    await websocket.send_bytes(WebSocketCodec.encode({"status": "ok", "type": "heartbeat"}, protocol))
                else:
                    await websocket.send_json({"status": "ok", "type": "heartbeat"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("ws_route_error", error=str(e))
    finally:
        await manager.disconnect(websocket)
