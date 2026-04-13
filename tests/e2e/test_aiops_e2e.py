from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.ml.aiops.autonomous_engine import AutonomousEngine
from src.ml.aiops.remediators import BaseRemediator
from src.ml.main import app

client = TestClient(app)


@pytest.fixture
def mock_health_dependencies():
    with (
        patch("src.ml.aiops.health_reporter.MlflowClient") as MockMlflowClient,
        patch("src.ml.aiops.health_reporter.PrometheusClient") as MockPrometheusClient,
        patch("src.ml.aiops.health_reporter.get_redis_client") as MockGetRedisClient,
    ):
        # Configure MLflow mock
        mock_mlflow = MockMlflowClient.return_value
        mock_mlflow.search_runs.return_value = [
            MagicMock(
                info=MagicMock(run_id="test-run"),
                data=MagicMock(tags={"drift_detected": "false", "stage": "production"}),
            )
        ]

        # Configure Prometheus mock
        mock_prom = MockPrometheusClient.return_value
        mock_prom.get_5xx_error_rate.return_value = 0.01
        mock_prom.get_p95_latency.return_value = 0.1
        mock_prom.prom.custom_query.return_value = [{"value": [0, "0.5"]}]

        # Configure Redis mock
        mock_redis = AsyncMock()
        MockGetRedisClient.return_value = mock_redis
        mock_redis.lrange.return_value = []

        yield {"mlflow": mock_mlflow, "prometheus": mock_prom, "redis": mock_redis}


def test_ml_health_endpoint(mock_health_dependencies):
    """
    Verify that the /ml/health endpoint returns a valid health report.
    """
    response = client.get("/ml/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "mlflow" in data
    assert "prometheus" in data
    assert "redis_anomalies" in data


@pytest.mark.asyncio
async def test_autonomous_engine_e2e_flow():
    """
    Verify the full flow from detection to remediation in AutonomousEngine.
    """
    config = {
        "prometheus_url": "http://mock-prometheus:9090",
        "api_service_name": "bsopt-api",
        "error_rate_threshold": 0.05,
    }

    with (
        patch("src.ml.aiops.autonomous_engine.PrometheusClient") as MockPrometheusClient,
        patch("src.ml.aiops.autonomous_engine.post_grafana_annotation") as MockNotify,
    ):
        mock_prom = MockPrometheusClient.return_value
        mock_prom.get_5xx_error_rate.return_value = 0.1  # Trigger anomaly
        mock_prom.get_p95_latency.return_value = 0.1

        # Mock a remediator
        class MockRemediator(BaseRemediator):
            def __init__(self):
                super().__init__("restart_service", ["high_error_rate"])
                self.remediate_mock = MagicMock(return_value=True)

            async def remediate(self, anomaly):
                return self.remediate_mock(anomaly)

        remediator = MockRemediator()
        engine = AutonomousEngine(config=config, remediators=[remediator])

        await engine.run_cycle()

        remediator.remediate_mock.assert_called_once()
        assert MockNotify.called