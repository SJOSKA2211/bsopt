from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import Optional
import structlog
from src.api.websockets.manager import manager, ConnectionMetadata, ProtocolType
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
    await manager.add_connection(websocket, symbol)
    
    try:
        while True:
            # Keep connection alive and wait for client disconnect
            # We don't need to process incoming messages for this one-way stream
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.remove_connection(websocket, symbol)
    except Exception as e:
        logger.error("ws_error", error=str(e))
        manager.remove_connection(websocket, symbol)
