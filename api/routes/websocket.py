import time

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from api.websockets.codec import WebSocketCodec
from api.websockets.manager import ProtocolType, manager

# In a real app, we would import auth dependencies here
# from api.dependencies import get_current_user

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
        from src.auth.auth import auth_service

        await auth_service.validate_token(token)
    except Exception:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return

    # 1. Initialize Metadata
    from api.websockets.manager import ConnectionMetadata

    websocket.metadata = ConnectionMetadata(protocol=protocol)

    # 2. Connect to manager
    await manager.connect(websocket)

    # 3. Initial subscription if provided
    if symbol:
        await manager.subscribe_to_symbol(websocket, symbol)

    try:
        while True:
            # 4. Dynamic Command Handling using fast binary paths
            data = await websocket.receive_bytes()
            msg = WebSocketCodec.decode(data, protocol)

            if not isinstance(msg, dict):
                continue

            action = msg.get("action")
            request_id = msg.get("request_id", "ws-" + str(int(time.time())))

            if action == "heartbeat":
                websocket.metadata.update_heartbeat()
                await websocket.send_bytes(
                    WebSocketCodec.encode({"status": "ok", "type": "heartbeat"}, protocol)
                )
                continue

            sym = msg.get("symbol")
            if not sym:
                continue

            if action == "subscribe":
                await manager.subscribe_to_symbol(websocket, sym)
                logger.info("ws_audit_subscribe", symbol=sym, request_id=request_id)
            elif action == "unsubscribe":
                await manager.unsubscribe_from_symbol(websocket, sym)
                logger.info("ws_audit_unsubscribe", symbol=sym, request_id=request_id)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("ws_route_error", error=str(e))
    finally:
        await manager.disconnect(websocket)


@router.websocket("/ws/greeks")
async def greeks_ws(
    websocket: WebSocket,
    symbol: str | None = Query(None, description="Initial symbol to subscribe to"),
    protocol: ProtocolType = Query(ProtocolType.JSON),
    token: str = Query(None, description="Bearer token for authentication"),
):
    """
    WebSocket endpoint for real-time mathematical feature (Greeks) data.
    """
    if not token:
        await websocket.close(code=1008, reason="Authentication required")
        return

    try:
        from src.auth.auth import auth_service

        await auth_service.validate_token(token)
    except Exception:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return

    from api.websockets.manager import ConnectionMetadata

    websocket.metadata = ConnectionMetadata(protocol=protocol)

    await manager.connect(websocket)

    if symbol:
        await manager.subscribe_to_symbol(websocket, f"GREEKS:{symbol.upper()}")

    try:
        while True:
            data = await websocket.receive_bytes()
            msg = WebSocketCodec.decode(data, protocol)

            if not isinstance(msg, dict):
                continue

            action = msg.get("action")
            if action == "subscribe":
                sym = msg.get("symbol")
                if sym:
                    await manager.subscribe_to_symbol(websocket, f"GREEKS:{sym.upper()}")
            elif action == "unsubscribe":
                sym = msg.get("symbol")
                if sym:
                    await manager.unsubscribe_from_symbol(websocket, f"GREEKS:{sym.upper()}")
            elif action == "heartbeat":
                websocket.metadata.update_heartbeat()
                await websocket.send_bytes(
                    WebSocketCodec.encode({"status": "ok", "type": "heartbeat"}, protocol)
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("ws_greeks_route_error", error=str(e))
    finally:
        await manager.disconnect(websocket)