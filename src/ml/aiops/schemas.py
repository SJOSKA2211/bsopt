import msgspec
from typing import Optional, List

class MLflowStatus(msgspec.Struct):
    stage: str
    drift_detected: bool
    last_run_id: Optional[str] = None

class PrometheusMetrics(msgspec.Struct):
    error_rate_5xx: float
    p95_latency: float
    cpu_usage: float
    memory_usage: float

class RedisAnomaly(msgspec.Struct):
    timestamp: str
    description: str
    severity: str

class MLHealthReport(msgspec.Struct):
    status: str
    mlflow: MLflowStatus
    prometheus: PrometheusMetrics
    redis_anomalies: List[RedisAnomaly]
    timestamp: str
