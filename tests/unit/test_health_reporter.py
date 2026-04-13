import sys
from unittest.mock import MagicMock

# Mock flwr and psycopg2 before importing src.ml
sys.modules["flwr"] = MagicMock()
sys.modules["psycopg2"] = MagicMock()

from unittest.mock import AsyncMock, patch

import pytest

from src.ml.aiops.health_reporter import HealthReporter
from src.ml.aiops.schemas import MLHealthReport


@pytest.fixture
def mock_mlflow():
    with patch("src.ml.aiops.health_reporter.MlflowClient") as mock:
        yield mock


@pytest.fixture
def mock_prometheus():
    with patch("src.ml.aiops.health_reporter.PrometheusClient") as mock:
        yield mock


@pytest.fixture
def mock_redis():
    with patch("src.ml.aiops.health_reporter.get_redis_client") as mock:
        yield mock


@pytest.mark.asyncio
async def test_get_health_report(mock_mlflow, mock_prometheus, mock_redis):
    # Setup mocks
    mock_mlflow_instance = mock_mlflow.return_value
    mock_mlflow_instance.search_runs.return_value = [
        MagicMock(
            info=MagicMock(run_id="test-run-id"),
            data=MagicMock(tags={"drift_detected": "false", "stage": "production"}),
        )
    ]

    mock_prom_instance = mock_prometheus.return_value
    mock_prom_instance.get_5xx_error_rate.return_value = 0.01
    mock_prom_instance.get_p95_latency.return_value = 0.1
    mock_prom_instance.prom.custom_query.return_value = [{"value": [0, "0.5"]}]

    mock_redis_instance = AsyncMock()
    mock_redis.return_value = mock_redis_instance
    mock_redis_instance.lrange.return_value = [
        b'{"timestamp": "2026-03-30T10:00:00Z", "description": "Test anomaly", "severity": "low"}'
    ]

    reporter = HealthReporter(prometheus_url="http://localhost:9090")
    report = await reporter.get_health_report()

    assert isinstance(report, MLHealthReport)
    assert report.status == "healthy"
    assert report.mlflow.stage == "production"
    assert report.prometheus.error_rate_5xx == 0.01
    assert len(report.redis_anomalies) == 1
    assert report.redis_anomalies[0].description == "Test anomaly"


@pytest.mark.asyncio
async def test_get_health_report_degraded(mock_mlflow, mock_prometheus, mock_redis):
    # Setup mocks for degraded status
    mock_mlflow_instance = mock_mlflow.return_value
    mock_mlflow_instance.search_runs.return_value = [
        MagicMock(
            info=MagicMock(run_id="test-run-id"),
            data=MagicMock(tags={"drift_detected": "true", "stage": "production"}),
        )
    ]

    mock_prom_instance = mock_prometheus.return_value
    mock_prom_instance.get_5xx_error_rate.return_value = 0.01
    mock_prom_instance.get_p95_latency.return_value = 0.1
    mock_prom_instance.prom.custom_query.return_value = []

    mock_redis_instance = AsyncMock()
    mock_redis.return_value = mock_redis_instance
    mock_redis_instance.lrange.return_value = []

    reporter = HealthReporter(prometheus_url="http://localhost:9090")
    report = await reporter.get_health_report()

    assert report.status == "degraded"
    assert report.mlflow.drift_detected is True