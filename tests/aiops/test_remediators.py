import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.aiops.remediators import (
    ClearRedisCacheRemediator,
    RestartServiceRemediator,
    KernelTuningRemediator,
    RemediationPlanner
)

@pytest.mark.asyncio
async def test_clear_redis_remediator():
    remediator = ClearRedisCacheRemediator()
    
    with patch("redis.asyncio.from_url") as mock_redis:
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client
        
        success = await remediator.remediate({"type": "latency_spike"})
        assert success is True
        mock_client.flushdb.assert_called_once()

@pytest.mark.asyncio
async def test_restart_service_remediator():
    remediator = RestartServiceRemediator()
    
    with patch("src.aiops.docker_remediator.DockerRemediator") as mock_docker_cls:
        mock_docker = MagicMock()
        mock_docker.restart_service = AsyncMock(return_value=True)
        mock_docker_cls.return_value = mock_docker
        
        success = await remediator.remediate({"metrics": {"service": "test-service"}})
        assert success is True
        mock_docker.restart_service.assert_called_once_with("test-service")

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
