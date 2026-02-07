from typing import Any

import msgspec
from starlette.responses import Response

# Core msgspec encoder instance
_encoder = msgspec.json.Encoder()

class MsgspecJSONResponse(Response):
    """
    JSON response class using msgspec for efficient serialization.
    Optimized for high-throughput API endpoints.
    """
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """Encode content to bytes."""
        try:
            return _encoder.encode(content)
        except Exception:
            # Fallback for complex objects not handled by the encoder
            import json
            return json.dumps(content).encode("utf-8")

def get_msgspec_response(content: Any, status_code: int = 200) -> MsgspecJSONResponse:
    """Helper for generating msgspec-powered responses."""
    return MsgspecJSONResponse(content=content, status_code=status_code)