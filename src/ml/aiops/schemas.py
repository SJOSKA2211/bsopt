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
    request_count: int

class RedisAnomaly(msgspec.Struct):
    timestamp: str
    description: str
    severity: str

class RabbitMQStatus(msgspec.Struct):
    connected: bool
    queue_depths: dict[str, int]
    consumer_counts: dict[str, int]

class RedisStatus(msgspec.Struct):
    connected: bool
    memory_usage_bytes: int
    total_keys: int

class PostgresStatus(msgspec.Struct):
    connected: bool
    version: str
    active_connections: int
    hypertables: int
    compression_ratio: float
    job_count: int

class AuthStatus(msgspec.Struct):
    reachable: bool
    p95_latency: float
    auth_success_rate: float
    active_tokens: int

class APIStatus(msgspec.Struct):
    reachable: bool
    p95_latency: float
    error_rate_5xx: float
    request_count: int

class RemediationStatus(msgspec.Struct):
    name: str
    status: str  # idle, cooldown, active
    last_run: str

class GuardianStatus(msgspec.Struct):
    active: bool
    safe_mode: bool
    paused_features: List[str]

class MLHealthReport(msgspec.Struct):
    status: str
    mlflow: MLflowStatus
    prometheus: PrometheusMetrics
    redis_anomalies: List[RedisAnomaly]
    rabbitmq: RabbitMQStatus
    redis: RedisStatus
    postgres: PostgresStatus
    api: APIStatus
    auth: AuthStatus
    remediations: List[RemediationStatus]
    guardian: GuardianStatus
    timestamp: str
