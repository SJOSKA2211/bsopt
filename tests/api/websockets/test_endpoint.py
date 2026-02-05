from unittest.mock import AsyncMock, patch

import pytest

from src.api.routes.websocket import market_data_ws
from src.api.websockets.codec import ProtocolType


@pytest.mark.asyncio
async def test_market_data_ws_protocol_negotiation():
    # Mock WebSocket
    websocket = AsyncMock()
    # Mock receive_text to raise exception immediately to exit loop
    websocket.receive_text.side_effect = Exception("exit_loop")
    
    symbol = "AAPL"
    protocol = ProtocolType.PROTO
    
    with patch("src.api.routes.websocket.manager.connect", new_callable=AsyncMock) as mock_connect:
        try:
            await market_data_ws(websocket, symbol, protocol)
        except Exception as e:
            if str(e) != "exit_loop":
                raise e
        
        # Verify manager.connect was called
        mock_connect.assert_called_once_with(websocket, symbol)
        
        # Verify metadata attached
        assert hasattr(websocket, "metadata")
        assert websocket.metadata.protocol == ProtocolType.PROTO
        assert symbol in websocket.metadata.subscriptions

@pytest.mark.asyncio
async def test_market_data_ws_default_protocol():
    websocket = AsyncMock()
    websocket.receive_text.side_effect = Exception("exit_loop")
    
    symbol = "GOOG"
    protocol = ProtocolType.JSON
    
    with patch("src.api.routes.websocket.manager.connect", new_callable=AsyncMock) as mock_connect:
        try:
            await market_data_ws(websocket, symbol, protocol)
        except Exception as e:
            if str(e) != "exit_loop":
                raise e
        
        assert websocket.metadata.protocol == ProtocolType.JSON