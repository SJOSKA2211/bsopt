"""
Standardized Market Data Schemas for EquaFlow.
Optimized for high-performance serialization via msgspec.
"""

import msgspec

class MarketQuote(msgspec.Struct):
    """
    Unified Market Quote object.
    Used for Ticker Tape, Portfolio valuation, and ML feature extraction.
    """
    symbol: str
    last_price: float
    prev_close: float
    change: float
    pct_change: float
    volume: int | None = None
    timestamp: float | None = None
    provider: str | None = None
    market: str = "US"

    @classmethod
    def from_price_change(
        cls, 
        symbol: str, 
        price: float, 
        change: float, 
        volume: int | None = None, 
        market: str = "US",
        provider: str | None = None
    ) -> "MarketQuote":
        """Factory for normalizing price/change into a full quote."""
        prev_close = price - change
        pct_change = (change / prev_close * 100) if prev_close != 0 else 0.0
        return cls(
            symbol=symbol,
            last_price=price,
            prev_close=prev_close,
            change=change,
            pct_change=pct_change,
            volume=volume,
            provider=provider,
            market=market
        )

    def to_ticker(self) -> "TickerSchema":
        """Convert to frontend-friendly TickerSchema."""
        return TickerSchema(
            symbol=self.symbol,
            price=f"{self.last_price:.2f}",
            change=f"{self.change:+.2f}",
            percentChange=f"{self.pct_change:+.2f}%",
            up=self.change >= 0
        )

class TickerSchema(msgspec.Struct):
    """
    Optimized schema for the Ticker Tape API.
    """
    symbol: str
    price: str
    change: str
    percentChange: str
    up: bool
