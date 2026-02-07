from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


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

class BatchInferenceRequest(BaseModel):
    """Batch ML inference request."""
    requests: list[InferenceRequest]

class InferenceResponse(BaseModel):

    """ML inference response."""

    price: float = Field(..., description="Predicted option price")

    model_type: str = Field(..., description="Model used for prediction")

    latency_ms: float = Field(..., description="Inference latency in milliseconds")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

class BatchInferenceResponse(BaseModel):
    """Batch ML inference response."""
    predictions: list[InferenceResponse]
    total_latency_ms: float



class DriftMetrics(BaseModel):

    """Hourly drift metrics from materialized view."""

    model_id: str

    window_hour: datetime

    mae: float

    rmse: float

    prediction_count: int



class DriftMetricsResponse(BaseModel):

    """Response containing a list of drift metrics."""

    metrics: list[DriftMetrics]
