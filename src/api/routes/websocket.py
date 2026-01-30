from fastapi import (
    APIRouter, WebSocket, WebSocketDisconnect, Query,
    Depends, status, WebSocketException
)
from typing import Optional, Dict
import structlog
from src.api.websockets.manager import (
    manager, ConnectionMetadata, ProtocolType
)
from src.auth.security import validate_token
from jose import JWTError

logger = structlog.get_logger()
router = APIRouter()


async def get_current_user_ws(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
) -> Dict:
    """
    Authenticate WebSocket connection via query parameter token.
    """
    if not token:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    try:
        payload = await validate_token(token)
        return payload
    except JWTError as e:
        logger.warning("ws_auth_failed", error=str(e))
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)


@router.websocket("/ws/market-data")
async def market_data_ws(
    websocket: WebSocket,
    symbol: str = Query(..., description="Ticker symbol to subscribe to"),
    protocol: ProtocolType = Query(
        ProtocolType.JSON,
        description="Protocol: json, proto, msgpack"
    ),
    user: Dict = Depends(get_current_user_ws)
):
    """
    WebSocket endpoint for real-time market data.
    Negotiates protocol and manages subscriptions.
    """
    # 1. Attach metadata to the websocket object for the manager
    websocket.metadata = ConnectionMetadata(
        user_id=user.get("sub"),
        protocol=protocol
    )

    # 2. Register connection
    await manager.connect(websocket, symbol)
    websocket.metadata.subscriptions.add(symbol)

    try:
        while True:
            # 3. Handle incoming messages (e.g., heartbeat or subscription)
            # For now, we just keep the connection alive
            data = await websocket.receive_text()
            websocket.metadata.update_heartbeat()

            # Placeholder for handling command messages
            logger.debug(
                "ws_received",
                user_id=websocket.metadata.user_id,
                data=data
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
        logger.info(
            "ws_disconnected",
            user_id=websocket.metadata.user_id,
            symbol=symbol
        )
    except Exception as e:
        logger.error("ws_error", error=str(e))
        manager.disconnect(websocket, symbol)
