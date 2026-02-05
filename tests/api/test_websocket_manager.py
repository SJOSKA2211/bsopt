from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket

from src.api.websockets.manager import ConnectionManager


@pytest.mark.asyncio
async def test_connection_manager_connect():
    with patch("src.api.websockets.manager.redis") as mock_redis_module:
        mock_redis = MagicMock() # Not AsyncMock
        mock_pubsub = MagicMock() # Not AsyncMock
        mock_pubsub.subscribe = AsyncMock() # This IS awaited
        mock_redis.pubsub.return_value = mock_pubsub
        mock_redis_module.from_url.return_value = mock_redis
        
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)
        
        await manager.connect(mock_ws, "AAPL")
        
        assert "AAPL" in manager.active_connections
        assert mock_ws in manager.active_connections["AAPL"]
        assert mock_ws.accept.called
        assert mock_pubsub.subscribe.called
        assert mock_pubsub.subscribe.call_args[0][0] == "AAPL"
        
        # Second connect to same symbol should NOT call subscribe again
        mock_pubsub.subscribe.reset_mock()
        mock_ws2 = AsyncMock(spec=WebSocket)
        await manager.connect(mock_ws2, "AAPL")
        assert not mock_pubsub.subscribe.called

@pytest.mark.asyncio
async def test_connection_manager_broadcast():
    with patch("src.api.websockets.manager.redis") as mock_redis_module:
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        
        # Default protocol is JSON, which uses send_text in manager
        manager.active_connections["AAPL"] = [mock_ws1, mock_ws2]
        
        message = {"price": 150.0}
        await manager.broadcast_to_symbol("AAPL", message)
        
        # orjson dumps returns bytes, but codec decode it to utf-8 str for JSON
        from src.api.websockets.codec import ProtocolType, WebSocketCodec
        expected_text = WebSocketCodec.encode(message, ProtocolType.JSON)
        
        assert mock_ws1.send_text.called
        assert mock_ws1.send_text.call_args[0][0] == expected_text
        assert mock_ws2.send_text.called
        assert mock_ws2.send_text.call_args[0][0] == expected_text

@pytest.mark.asyncio
async def test_broadcast_no_connections():
    with patch("src.api.websockets.manager.redis") as mock_redis_module:
        manager = ConnectionManager()
        # Symbol exists but list is empty
        manager.active_connections["AAPL"] = []
        await manager.broadcast_to_symbol("AAPL", {"msg": "test"})

@pytest.mark.asyncio
async def test_broadcast_unknown_symbol():
    with patch("src.api.websockets.manager.redis") as mock_redis_module:
        manager = ConnectionManager()
        # Should return early without error
        await manager.broadcast_to_symbol("UNKNOWN", {"msg": "test"})

@pytest.mark.asyncio
async def test_connection_manager_disconnect():
    with patch("src.api.websockets.manager.redis") as mock_redis_module:
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)
        manager.active_connections["AAPL"] = [mock_ws]
        
        manager.disconnect(mock_ws, "AAPL")
        assert "AAPL" not in manager.active_connections
        
        # Disconnect unknown symbol
        manager.disconnect(mock_ws, "UNKNOWN")
        
        # Disconnect from symbol where ws is not present
        manager.active_connections["GOOG"] = []
        manager.disconnect(mock_ws, "GOOG")

@pytest.mark.asyncio
async def test_disconnect_multiple_clients():
    with patch("src.api.websockets.manager.redis") as mock_redis_module:
        manager = ConnectionManager()
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws2 = AsyncMock(spec=WebSocket)
        manager.active_connections["AAPL"] = [mock_ws1, mock_ws2]
        
        # Disconnect one, list should remain
        manager.disconnect(mock_ws1, "AAPL")
        assert "AAPL" in manager.active_connections
        assert len(manager.active_connections["AAPL"]) == 1
        
        # Disconnect last, list should be deleted
        manager.disconnect(mock_ws2, "AAPL")
        assert "AAPL" not in manager.active_connections

@pytest.mark.asyncio
async def test_broadcast_with_exception():
    with patch("src.api.websockets.manager.redis") as mock_redis_module:
        manager = ConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.send_text.side_effect = Exception("Send failed")
        
        manager.active_connections["AAPL"] = [mock_ws]
        # Should not raise exception because of return_exceptions=True
        await manager.broadcast_to_symbol("AAPL", {"msg": "test"})
        assert mock_ws.send_text.called
