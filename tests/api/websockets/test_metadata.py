import pytest
from datetime import datetime
import json
import msgpack
from src.api.websockets.manager import ConnectionMetadata, ProtocolType
from src.protos.market_data_pb2 import TickerUpdate
from src.api.websockets.codec import WebSocketCodec

def test_connection_metadata_initialization():
    metadata = ConnectionMetadata(user_id="user123", protocol=ProtocolType.JSON)
    assert metadata.user_id == "user123"
    assert metadata.protocol == ProtocolType.JSON
    assert isinstance(metadata.subscriptions, set)
    assert len(metadata.subscriptions) == 0
    assert isinstance(metadata.last_heartbeat, datetime)

def test_connection_metadata_update_heartbeat():
    metadata = ConnectionMetadata()
    initial_heartbeat = metadata.last_heartbeat
    # Ensure some time difference if executed extremely fast
    import time
    time.sleep(0.001)
    metadata.update_heartbeat()
    assert metadata.last_heartbeat > initial_heartbeat

def test_codec_json_serialization():
    data = {"symbol": "AAPL", "price": 150.0}
    encoded = WebSocketCodec.encode(data, ProtocolType.JSON)
    # JSON encoding usually returns str for websockets (text frame)
    assert isinstance(encoded, str)
    # orjson is compact, so check without spaces
    assert '"symbol":"AAPL"' in encoded
    
    decoded = WebSocketCodec.decode(encoded, ProtocolType.JSON)
    assert decoded["symbol"] == "AAPL"

def test_codec_proto_to_json_conversion():
    ticker = TickerUpdate(symbol="AAPL", price=150.0)
    encoded = WebSocketCodec.encode(ticker, ProtocolType.JSON)
    assert isinstance(encoded, str)
    assert '"symbol":"AAPL"' in encoded

def test_codec_proto_serialization():
    ticker = TickerUpdate()
    ticker.symbol = "AAPL"
    ticker.price = 150.0 
    
    encoded = WebSocketCodec.encode(ticker, ProtocolType.PROTO)
    assert isinstance(encoded, bytes)
    
    # Verify we can decode it back manually to check validity
    decoded_ticker = TickerUpdate()
    decoded_ticker.ParseFromString(encoded)
    assert decoded_ticker.symbol == "AAPL"
    assert decoded_ticker.price == 150.0

def test_codec_proto_invalid_input():
    with pytest.raises(ValueError):
        WebSocketCodec.encode({"not": "proto"}, ProtocolType.PROTO)

def test_codec_proto_decoding_not_implemented():
    with pytest.raises(NotImplementedError):
        WebSocketCodec.decode(b"somebytes", ProtocolType.PROTO)

def test_codec_msgpack_serialization():
    data = {"symbol": "AAPL", "price": 150.0}
    encoded = WebSocketCodec.encode(data, ProtocolType.MSGPACK)
    assert isinstance(encoded, bytes)
    
    decoded = WebSocketCodec.decode(encoded, ProtocolType.MSGPACK)
    assert decoded["symbol"] == "AAPL"

def test_codec_invalid_protocol():
    with pytest.raises(ValueError):
        WebSocketCodec.encode({}, "invalid_protocol")

def test_codec_decode_invalid_protocol():
    with pytest.raises(ValueError):
        WebSocketCodec.decode(b"{}", "invalid_protocol")