from typing import List, Optional
from datetime import datetime


class VolatilityPoint:
    def __init__(self, expiry: datetime, strike: float, implied_volatility: float):
        self.expiry = expiry
        self.strike = strike
        self.implied_volatility = implied_volatility

async def get_vol_surface(
    underlying: str,
    expiry_range: Optional[int] = 90
) -> List[VolatilityPoint]:
    """Dummy resolver for fetching implied volatility surface."""
    print(f"Dummy get_vol_surface called for {underlying}, {expiry_range}")
    return [VolatilityPoint(expiry=datetime.now(), strike=100.0, implied_volatility=0.25)]
