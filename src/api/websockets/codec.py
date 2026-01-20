from enum import Enum
from typing import Any, Union
import orjson
import msgpack
from google.protobuf.message import Message
from google.protobuf.json_format import MessageToDict

class ProtocolType(str, Enum):
    JSON = "json"
    PROTO = "proto"
    MSGPACK = "msgpack"

class WebSocketCodec:
    @staticmethod
    def encode(data: Any, protocol: ProtocolType) -> Union[str, bytes]:
        if protocol == ProtocolType.JSON:
            if isinstance(data, Message):
                data = MessageToDict(data, preserving_proto_field_name=True)
            return orjson.dumps(data).decode('utf-8')
        elif protocol == ProtocolType.MSGPACK:
            return msgpack.packb(data)
        elif protocol == ProtocolType.PROTO:
            if not isinstance(data, Message):
                raise ValueError("Data must be a Protobuf Message for PROTO protocol")
            return data.SerializeToString()
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

    @staticmethod
    def decode(data: Union[str, bytes], protocol: ProtocolType) -> Any:
        if protocol == ProtocolType.JSON:
            return orjson.loads(data)
        elif protocol == ProtocolType.MSGPACK:
            return msgpack.unpackb(data)
        elif protocol == ProtocolType.PROTO:
            raise NotImplementedError("Generic proto decoding not supported in Codec helper")
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
