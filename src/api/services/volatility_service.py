import structlog
from typing import List, Optional
from datetime import datetime
import strawberry

logger = structlog.get_logger(__name__)

@strawberry.type
class VolatilitySlice:
    strike: float
    implied_vol: float

@strawberry.type
class VolatilitySurface:
    underlying: str
    slices: List[VolatilitySlice]

async def get_vol_surface(underlying: str, expiry_range: Optional[List[datetime]] = None) -> VolatilitySurface:
    """Dummy resolver for volatility surface."""
    logger.debug("dummy_vol_surface_fetch", underlying=underlying)
    return VolatilitySurface(
        underlying=underlying,
        slices=[VolatilitySlice(strike=100.0, implied_vol=0.2)]
    )
