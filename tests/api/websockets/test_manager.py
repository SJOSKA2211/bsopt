import pytest
from unittest.mock import AsyncMock, Mock
from src.api.websockets.manager import ConnectionManager, ConnectionMetadata, ProtocolType
from src.protos.market_data_pb2 import TickerUpdate

@pytest.fixture
def manager():
    mgr = ConnectionManager()
    mgr.redis = Mock()
    mgr.pubsub = Mock()
    mgr.pubsub.subscribe = AsyncMock()
    mgr.pubsub.unsubscribe = AsyncMock()
    return mgr

@pytest.mark.asyncio
async def test_connect_disconnect(manager):
    ws = AsyncMock()
    # Mocking metadata attachment
    ws.metadata = ConnectionMetadata()
    
    await manager.connect(ws, "AAPL")
    assert "AAPL" in manager.active_connections
    assert ws in manager.active_connections["AAPL"]
    manager.pubsub.subscribe.assert_called_with("AAPL")
    
    manager.disconnect(ws, "AAPL")
    assert "AAPL" not in manager.active_connections

@pytest.mark.asyncio
async def test_broadcast_mixed_protocols(manager):
    ws_json = AsyncMock()
    ws_json.metadata = ConnectionMetadata(protocol=ProtocolType.JSON)
    
    ws_proto = AsyncMock()
    ws_proto.metadata = ConnectionMetadata(protocol=ProtocolType.PROTO)
    
    manager.active_connections["AAPL"] = [ws_json, ws_proto]
    
    # Input data is Protobuf message
    data = TickerUpdate(symbol="AAPL", price=150.0)
    
    await manager.broadcast_to_symbol("AAPL", data)
    
    # Verify JSON client received text
    # Codec returns str for JSON, so we expect send_text
    assert ws_json.send_text.called
    sent_json = ws_json.send_text.call_args[0][0]
    assert '"symbol":"AAPL"' in sent_json
    assert '"price":150.0' in sent_json
    
    # Verify Proto client received bytes
    assert ws_proto.send_bytes.called
    ws_proto.send_bytes.assert_called_with(data.SerializeToString())

@pytest.mark.asyncio
async def test_broadcast_dict_input(manager):
    # Test backward compatibility if input is dict
    ws_json = AsyncMock()
    ws_json.metadata = ConnectionMetadata(protocol=ProtocolType.JSON)
    manager.active_connections["AAPL"] = [ws_json]
    
    data = {"symbol": "AAPL", "price": 150.0}
    await manager.broadcast_to_symbol("AAPL", data)
    
    assert ws_json.send_text.called
    sent_json = ws_json.send_text.call_args[0][0]
    assert '"symbol":"AAPL"' in sent_json

@pytest.mark.asyncio
async def test_broadcast_encode_error(manager):
    ws_proto = AsyncMock()
    ws_proto.metadata = ConnectionMetadata(protocol=ProtocolType.PROTO)
    manager.active_connections["AAPL"] = [ws_proto]
    
    # Send invalid data for Proto (e.g. dict when it expects Message)
    data = {"not": "proto"}
    
    # Should log error and not crash
    await manager.broadcast_to_symbol("AAPL", data)
    
    assert not ws_proto.send_bytes.called

@pytest.mark.asyncio
async def test_init_redis_connection():
    from unittest.mock import patch
    with patch("src.api.websockets.manager.redis.from_url") as mock_redis:
        mgr = ConnectionManager()
        mock_redis.assert_called_once()
        assert mgr.redis == mock_redis.return_value

@pytest.mark.asyncio
async def test_broadcast_unknown_symbol(manager):
    # Should not raise error
    await manager.broadcast_to_symbol("UNKNOWN", {})

@pytest.mark.asyncio
async def test_disconnect_unknown(manager):
    ws = AsyncMock()
    # Disconnect a socket that wasn't connected
    manager.disconnect(ws, "AAPL")
    # Should handle gracefully (log and exit)
    assert "AAPL" not in manager.active_connections


