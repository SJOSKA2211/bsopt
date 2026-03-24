from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ml.aiops.remediators import (
    ClearRedisCacheRemediator,
    DatabasePoolRemediator,
    KernelTuningRemediator,
    RabbitMQCongestionRemediator,
    RemediationPlanner,
    RestartServiceRemediator,
)

@pytest.mark.asyncio
async def test_clear_redis_remediator():
    remediator = ClearRedisCacheRemediator()

    with patch("src.ml.aiops.remediators.redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_client.flushdb = AsyncMock()
        mock_from_url.return_value = mock_client

        success = await remediator.remediate({"type": "latency_spike"})
        assert success is True
        mock_client.flushdb.assert_called_once()

@pytest.mark.asyncio
async def test_restart_service_remediator_success():
    remediator = RestartServiceRemediator()

    with patch("src.ml.aiops.docker_remediator.DockerRemediator") as mock_docker_cls:
        mock_docker = MagicMock()
        mock_docker.restart_service = AsyncMock(return_value=True)
        mock_docker_cls.return_value = mock_docker

        success = await remediator.remediate({"metrics": {"service": "api"}})
        assert success is True
        mock_docker.restart_service.assert_called_once_with("api")

@pytest.mark.asyncio
async def test_docker_remediator_invalid():
    from src.ml.aiops.docker_remediator import DockerRemediator

    remediator = DockerRemediator()

    # Should reject non-allowlisted service
    assert remediator._validate_service("malicious-container") is False
    # Should reject malformed service
    assert remediator._validate_service("api; rm -rf /") is False
    # Should accept allowlisted service
    assert remediator._validate_service("worker") is True

@pytest.mark.asyncio
async def test_docker_remediator_scale_bounds():
    from src.ml.aiops.docker_remediator import DockerRemediator

    remediator = DockerRemediator()
    remediator.loop = MagicMock()
    remediator.executor = MagicMock()

    # Scale too high should fail
    assert await remediator.scale_service("api", 100) is False
    # Scale too low should fail
    assert await remediator.scale_service("api", 0) is False

@pytest.mark.asyncio
async def test_kernel_tuning_remediator():
    remediator = KernelTuningRemediator()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"OK", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        success = await remediator.remediate({})
        assert success is True
        mock_exec.assert_called_once()

@pytest.mark.asyncio
async def test_remediation_planner():
    planner = RemediationPlanner()

    # Should suggest clear_cache for latency_spike
    anomaly = {"type": "latency_spike"}
    actions = planner.plan(anomaly)

    action_names = [a.name for a in actions]
    assert "clear_cache" in action_names
    assert "restart_service" in action_names
    assert "kernel_tuning" in action_names

    # Should suggest retrain_model for model_drift
    anomaly_drift = {"type": "model_drift"}
    actions_drift = planner.plan(anomaly_drift)
    assert "retrain_model" in [a.name for a in actions_drift]

@pytest.mark.asyncio
async def test_db_pool_remediator_success():
    remediator = DatabasePoolRemediator()

    with patch("src.database.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.dispose = MagicMock()
        mock_get_engine.return_value = mock_engine

        success = await remediator.remediate(
            {"type": "db_pool_exhaustion", "metrics": {"pool_utilization": 0.5}}
        )
        assert success is True
        mock_engine.dispose.assert_called_once()

@pytest.mark.asyncio
async def test_db_pool_remediator_critical_pressure():
    remediator = DatabasePoolRemediator()

    with patch("src.database.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.dispose = MagicMock()
        mock_get_engine.return_value = mock_engine

        success = await remediator.remediate(
            {"type": "db_pool_exhaustion", "metrics": {"pool_utilization": 0.95}}
        )
        assert success is True

@pytest.mark.asyncio
async def test_db_pool_remediator_failure():
    remediator = DatabasePoolRemediator()

    with patch("src.database.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.dispose.side_effect = Exception("Connection refused")
        mock_get_engine.return_value = mock_engine

        success = await remediator.remediate({"type": "db_pool_exhaustion"})
        assert success is False

@pytest.mark.asyncio
async def test_rabbitmq_congestion_remediator_purge_dlq():
    remediator = RabbitMQCongestionRemediator()

    with patch("aio_pika.connect_robust") as mock_connect:
        mock_channel = AsyncMock()
        mock_queue = AsyncMock()
        mock_channel.declare_queue.return_value = mock_queue
        mock_conn = AsyncMock()
        mock_conn.channel.return_value = mock_channel
        mock_connect.return_value = mock_conn

        success = await remediator.remediate(
            {"metrics": {"queue": "ml_tasks", "suggested_action": "purge_dlq"}}
        )
        assert success is True
        mock_channel.declare_queue.assert_called_once_with("ml_tasks.dlq", passive=True)
        mock_queue.purge.assert_called_once()

@pytest.mark.asyncio
async def test_rabbitmq_congestion_remediator_increase_prefetch():
    remediator = RabbitMQCongestionRemediator()

    with patch("aio_pika.connect_robust") as mock_connect:
        mock_channel = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.channel.return_value = mock_channel
        mock_connect.return_value = mock_conn

        success = await remediator.remediate(
            {"metrics": {"queue": "pricing", "suggested_action": "increase_prefetch"}}
        )
        assert success is True
        mock_channel.set_qos.assert_called_once_with(prefetch_count=50)

@pytest.mark.asyncio
async def test_rabbitmq_congestion_remediator_restart_consumers():
    remediator = RabbitMQCongestionRemediator()

    with patch("aio_pika.connect_robust") as mock_connect:
        mock_channel = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.channel.return_value = mock_channel
        mock_connect.return_value = mock_conn

        with patch("src.ml.aiops.docker_remediator.DockerRemediator") as mock_docker_cls:
            mock_docker = MagicMock()
            mock_docker.restart_service = AsyncMock(return_value=True)
            mock_docker_cls.return_value = mock_docker

            success = await remediator.remediate(
                {"metrics": {"queue": "scraper", "suggested_action": "restart_consumers"}}
            )
            assert success is True
            mock_docker.restart_service.assert_called_once_with("worker")

@pytest.mark.asyncio
async def test_rabbitmq_congestion_remediator_forbidden():
    remediator = RabbitMQCongestionRemediator()

    # Forbidden queue
    success = await remediator.remediate(
        {"metrics": {"queue": "unknown_queue", "suggested_action": "purge_dlq"}}
    )
    assert success is False

    # Forbidden action
    success = await remediator.remediate(
        {"metrics": {"queue": "default", "suggested_action": "delete_everything"}}
    )
    assert success is False

@pytest.mark.asyncio
async def test_planner_includes_new_remediators():
    planner = RemediationPlanner()

    # DatabasePoolRemediator should respond to db_pool_exhaustion
    db_anomaly = {"type": "db_pool_exhaustion"}
    db_actions = planner.plan(db_anomaly)
    assert "db_pool_recovery" in [a.name for a in db_actions]

    # RabbitMQCongestionRemediator should respond to queue_backpressure
    rmq_anomaly = {"type": "queue_backpressure"}
    rmq_actions = planner.plan(rmq_anomaly)
    assert "rabbitmq_congestion" in [a.name for a in rmq_actions]
