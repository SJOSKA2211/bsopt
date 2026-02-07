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
    protocol: ProtocolType = Query(ProtocolType.JSON, description="Protocol: json, proto, msgpack"),
):
    """
    WebSocket endpoint for real-time market data.
    """
    # Optimized: Direct registration without intermediate metadata object overhead
    await manager.connect(websocket, symbol)
    
    # Set protocol on the websocket object itself for the manager to read
    if not hasattr(websocket, "metadata"):
         from src.api.websockets.manager import ConnectionMetadata
         websocket.metadata = ConnectionMetadata(protocol=protocol)
    else:
         websocket.metadata.protocol = protocol

    try:
        while True:
            # Keep connection alive and wait for client disconnect
            # We don't need to process incoming messages for this one-way stream
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
    except Exception as e:
        logger.error("ws_error", error=str(e))
        manager.disconnect(websocket, symbol)
