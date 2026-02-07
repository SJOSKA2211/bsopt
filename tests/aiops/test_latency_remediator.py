from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.aiops.latency_remediator import LatencyRemediator


@pytest.fixture
def remediator():
    return LatencyRemediator(threshold_ms=10.0)

@pytest.mark.asyncio
async def test_latency_remediator_init(remediator):
    assert remediator.threshold_ms == 10.0
    assert remediator.last_remediation == 0

@pytest.mark.asyncio
@patch("src.aiops.latency_remediator.get_obs_client")
async def test_query_p95_latency_success(mock_get_client, remediator):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "result": [
                {"value": [0, "0.015"]} # 15ms
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_get_client.return_value = mock_client
    
    latency = await remediator._query_p95_latency()
    assert latency == 15.0

@pytest.mark.asyncio
@patch("src.aiops.latency_remediator.get_obs_client")
async def test_query_p95_latency_failure(mock_get_client, remediator):
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("Prometheus down")
    mock_get_client.return_value = mock_client
    
    latency = await remediator._query_p95_latency()
    assert latency == 0.0

@pytest.mark.asyncio
async def test_check_and_remediate_cooldown(remediator):
    import time
    remediator.last_remediation = time.time()
    # Should return early
    with patch.object(remediator, "_query_p95_latency") as mock_query:
        await remediator.check_and_remediate()
        mock_query.assert_not_called()

@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_trigger_kernel_tuning_success(mock_exec, remediator):
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"output", b"")
    mock_proc.returncode = 0
    mock_exec.return_value = mock_proc
    
    await remediator._trigger_kernel_tuning()
    mock_exec.assert_called_once_with(
        "sudo", "/app/scripts/optimize_kernel.sh",
        stdout=-1, stderr=-1
    )

@pytest.mark.asyncio
@patch("asyncio.create_subprocess_exec")
async def test_trigger_kernel_tuning_failure(mock_exec, remediator):
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"", b"error")
    mock_proc.returncode = 1
    mock_exec.return_value = mock_proc
    
    # Should log error but not raise
    await remediator._trigger_kernel_tuning()
    mock_exec.assert_called_once()
