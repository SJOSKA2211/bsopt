from typing import Any

import msgspec
from starlette.responses import Response

# 🚀 SINGULARITY: High-performance msgspec encoder
_encoder = msgspec.json.Encoder()

class MsgspecJSONResponse(Response):
    """
    SOTA: Ultra-fast JSON response using msgspec.
    Bypasses standard Python dict-to-json conversion for direct byte serialization.
    """
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """🚀 SINGULARITY: Direct byte serialization."""
        try:
            return _encoder.encode(content)
        except Exception:
            # Fallback for non-serializable objects (though msgspec is robust)
            import json
            return json.dumps(content).encode("utf-8")

def get_msgspec_response(content: Any, status_code: int = 200) -> MsgspecJSONResponse:
    """Helper for generating msgspec-powered responses."""
    return MsgspecJSONResponse(content=content, status_code=status_code)