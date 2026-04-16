from datetime import UTC, datetime

import msgspec
from pydantic import BaseModel, ConfigDict, Field


class InferenceRequest(BaseModel):
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
    requests: list[InferenceRequest]


class InferenceResponse(msgspec.Struct, frozen=True):
    price: float
    model_type: str
    latency_ms: float
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.now(UTC))


class BatchInferenceResponse(msgspec.Struct, frozen=True):
    predictions: list[InferenceResponse]
    total_latency_ms: float


class ComparisonMetrics(msgspec.Struct, frozen=True):
    user_pnl: float
    ai_pnl: float
    user_sharpe: float
    ai_sharpe: float
    user_win_rate: float
    ai_win_rate: float


class DriftMetrics(msgspec.Struct, frozen=True):
    model_id: str
    window_hour: datetime
    mae: float
    rmse: float
    prediction_count: int


class DriftMetricsResponse(msgspec.Struct, frozen=True):
    metrics: list[DriftMetrics]
