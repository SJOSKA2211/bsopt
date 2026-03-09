from datetime import datetime

import msgspec
from pydantic import BaseModel, ConfigDict, Field


class InferenceRequest(BaseModel):
    """ML inference request (Pydantic for Request Validation)."""

    underlying_price: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    time_to_expiry: float = Field(..., gt=0)
    is_call: int = Field(..., description="1 for call, 0 for put")
    moneyness: float = Field(..., gt=0)
    log_moneyness: float = Field(...)
    sqrt_time_to_expiry: float = Field(..., gt=0)
    days_to_expiry: float = Field(..., gt=0)
    implied_volatility: float | None = Field(None, ge=0)

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
                "implied_volatility": 0.2,
            }
        }
    )


class BatchInferenceRequest(BaseModel):
    """Batch ML inference request."""

    requests: list[InferenceRequest]


class InferenceResponse(msgspec.Struct):
    """ML inference response (msgspec for Response Speed)."""

    price: float
    model_type: str
    latency_ms: float
    timestamp: datetime = msgspec.field(default_factory=datetime.utcnow)


class BatchInferenceResponse(msgspec.Struct):
    """Batch ML inference response."""

    predictions: list[InferenceResponse]
    total_latency_ms: float


class DriftMetrics(msgspec.Struct):
    """Hourly drift metrics from materialized view."""

    model_id: str
    window_hour: datetime
    mae: float
    rmse: float
    prediction_count: int


class DriftMetricsResponse(msgspec.Struct):
    """Response containing a list of drift metrics."""

    metrics: list[DriftMetrics]
