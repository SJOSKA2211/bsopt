from enum import StrEnum
from typing import Any

import msgspec
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message


class ProtocolType(StrEnum):
    JSON = "json"
    PROTO = "proto"
    MSGPACK = "msgpack"


class WebSocketCodec:
    """
    Ultra-high-performance codec for multi-protocol serialization.
    FUSED: Uses msgspec for binary speed.
    """

    _msgpack_encoder = msgspec.msgpack.Encoder()
    _msgpack_decoder = msgspec.msgpack.Decoder()
    _json_encoder = msgspec.json.Encoder()
    _json_decoder = msgspec.json.Decoder()

    @staticmethod
    def encode(data: Any, protocol: ProtocolType) -> bytes:
        if protocol == ProtocolType.JSON:
            if isinstance(data, Message):
                data = MessageToDict(data, preserving_proto_field_name=True)
            return WebSocketCodec._json_encoder.encode(data)
        if protocol == ProtocolType.MSGPACK:
            return WebSocketCodec._msgpack_encoder.encode(data)
        if protocol == ProtocolType.PROTO:
            if not isinstance(data, Message):
                raise ValueError("Data must be a Protobuf Message for PROTO protocol")
            return data.SerializeToString()
        raise ValueError(f"Unsupported protocol: {protocol}")

    @staticmethod
    def decode(data: str | bytes, protocol: ProtocolType, message_type: Any | None = None) -> Any:
        if protocol == ProtocolType.JSON:
            return WebSocketCodec._json_decoder.decode(data)
        if protocol == ProtocolType.MSGPACK:
            # OPTIMIZED: Use pre-allocated msgspec decoder
            return WebSocketCodec._msgpack_decoder.decode(data)
        if protocol == ProtocolType.PROTO:
            # High-performance binary decoding
            if message_type is None:
                raise ValueError("message_type required for PROTO decoding")
            message = message_type()
            message.ParseFromString(data)
            return message
        raise ValueError(f"Unsupported protocol: {protocol}")
