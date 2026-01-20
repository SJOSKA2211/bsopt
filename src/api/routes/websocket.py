from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import Optional
import structlog
from src.api.websockets.manager import manager, ConnectionMetadata, ProtocolType
from src.security.auth import get_current_user

logger = structlog.get_logger()
router = APIRouter()

@router.websocket("/ws/market-data")
async def market_data_ws(
    websocket: WebSocket,
    symbol: str = Query(..., description="Ticker symbol to subscribe to"),
    protocol: ProtocolType = Query(ProtocolType.JSON, description="Protocol: json, proto, msgpack"),
    user = Depends(get_current_user)
):
    """
    WebSocket endpoint for real-time market data.
    Negotiates protocol and manages subscriptions.
    """
    websocket.metadata = ConnectionMetadata(
        user_id=str(user.id),
        protocol=protocol
    )
    
    await manager.connect(websocket, symbol)
    websocket.metadata.subscriptions.add(symbol)
    
    try:
        while True:
            # Keep connection alive, actual data push is handled by manager
            data = await websocket.receive_text()
            websocket.metadata.update_heartbeat()
            logger.debug("ws_received", user_id=websocket.metadata.user_id, data=data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
        logger.info("ws_disconnected", user_id=websocket.metadata.user_id, symbol=symbol)
    except Exception as e:
        logger.error("ws_error", error=str(e))
        manager.disconnect(websocket, symbol)
