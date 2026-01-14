from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal
from datetime import datetime, timezone

class InferenceRequest(BaseModel):
    """ML inference request."""
    underlying_price: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    time_to_expiry: float = Field(..., gt=0)
    is_call: int = Field(..., description="1 for call, 0 for put")
    moneyness: float = Field(..., gt=0)
    log_moneyness: float = Field(...)
    sqrt_time_to_expiry: float = Field(..., gt=0)
    days_to_expiry: float = Field(..., gt=0)
    implied_volatility: float = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "underlying_price": 100.0,
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "is_call": 1,
                "moneyness": 1.0,
                "log_moneyness": 0.0,
                "sqrt_time_to_expiry": 1.0,
                "days_to_expiry": 365.0,
                "implied_volatility": 0.2
            }
        }
    )

class InferenceResponse(BaseModel):
    """ML inference response."""
    price: float = Field(..., description="Predicted option price")
    model_type: str = Field(..., description="Model used for prediction")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))