import msgspec
from typing import List, Optional

class MarketData(msgspec.Struct):
    """🚀 SINGULARITY: Binary-level schema for high-throughput market data."""
    symbol: str
    spot: float
    strike: float
    maturity: float
    volatility: float
    rate: float
    is_call: bool
    timestamp: float

class OptionsDataValidator:
    """
    SOTA: High-performance data validator using msgspec.
    Bypasses pandas overhead by validating raw JSON/MessagePack bytes.
    """
    def __init__(self, min_samples: int):
        self.min_samples = min_samples
        self._decoder = msgspec.json.Decoder(List[MarketData])

    def validate_raw(self, data: bytes) -> bool:
        """Ultra-fast validation of raw bytes."""
        try:
            # SOTA: msgspec validates while decoding in a single pass
            records = self._decoder.decode(data)
            if len(records) < self.min_samples:
                return False
            # Check for negative strikes (logical validation)
            for r in records:
                if r.strike < 0: return False
            return True
        except Exception:
            return False

    def validate(self, df: any) -> any:
        # Compatibility wrapper for legacy code
        from typing import NamedTuple
        class ValidationReport(NamedTuple):
            passed: bool
            errors: list[str] = []
            
        if hasattr(df, "empty") and df.empty:
            return ValidationReport(passed=False, errors=["DataFrame is empty"])
        
        return ValidationReport(passed=True)