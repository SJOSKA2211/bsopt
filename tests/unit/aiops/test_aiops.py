import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.aiops.remediators import (
    RemediationPlanner,
    RestartServiceRemediator,
    RetrainModelRemediator,
)
from src.aiops.self_healing_orchestrator import SelfHealingOrchestrator
from src.aiops.timeseries_anomaly_detector import TimeSeriesAnomalyDetector


@pytest.fixture
def sample_metrics():
    return pd.DataFrame({
        "latency": np.random.normal(50, 5, 100),
        "errors": np.random.normal(0.01, 0.001, 100),
        "cpu": np.random.normal(30, 10, 100)
    })

@pytest.fixture
def detector():
    d = TimeSeriesAnomalyDetector()
    # Train it so it's fitted
    d.train(pd.DataFrame({"latency": [50, 51, 49, 50, 52], "errors": [0, 0, 0, 0, 0], "cpu": [30, 31, 29, 30, 32]}))
    return d

@pytest.fixture
def orchestrator(detector):
    remediators = [RestartServiceRemediator(), RetrainModelRemediator()]
    return SelfHealingOrchestrator(detector, remediators)

@pytest.mark.asyncio
async def test_detector_detect(detector):
    # Test normal data
    normal_data = pd.DataFrame({"latency": [50.0], "errors": [0.0], "cpu": [30.0]})
    anomalies = detector.detect(normal_data)
    assert len(anomalies) == 0
    
    # Test anomaly (Isolation Forest needs more points to reliably detect an outlier sometimes, but let's try)
    extreme_data = pd.DataFrame({"latency": [500.0], "errors": [1.0], "cpu": [100.0]})
    # Mock model predict since Isolation Forest is stochastic
    with patch.object(detector.model, "predict", return_value=np.array([-1])):
        with patch.object(detector.model, "decision_function", return_value=np.array([-0.5])):
            anomalies = detector.detect(extreme_data)
            assert len(anomalies) == 1
            assert anomalies[0]["score"] < 0

def test_remediation_planner():
    remediators = [RestartServiceRemediator(), RetrainModelRemediator()]
    planner = RemediationPlanner(remediators)
    
    # Test planning for latency spike
    actions = planner.plan({"type": "latency_spike"})
    assert any(a.name == "restart_service" for a in actions)
    
    # Test planning for model drift
    actions = planner.plan({"type": "model_drift"})
    assert any(a.name == "retrain_model" for a in actions)

@pytest.mark.asyncio
async def test_orchestrator_run_cycle(orchestrator, sample_metrics):
    # Mock detector to return one anomaly
    orchestrator.detector.detect = MagicMock(return_value=[{"type": "latency_spike", "metrics": {"service": "api"}}])
    
    # Mock remediator to avoid actual sleep/tasks
    with patch.object(RestartServiceRemediator, "remediate", new_callable=AsyncMock) as mock_rem:
        orchestrator.planner.plan = MagicMock(return_value=[orchestrator.remediators[0]])
        
        await orchestrator.run_cycle(sample_metrics)
        
        orchestrator.detector.detect.assert_called_once()
        mock_rem.assert_called_once()

def test_analyze_drift(orchestrator, sample_metrics):
    # First call initializes baseline
    anomalies = orchestrator._analyze_drift(sample_metrics)
    assert len(anomalies) == 0
    assert orchestrator.reference_data is not None
    
    # Second call with shifted data
    shifted_metrics = sample_metrics.copy()
    shifted_metrics["latency"] += 100 # Large shift
    
    with patch("src.aiops.self_healing_orchestrator.calculate_psi", return_value=0.5):
        with patch("src.aiops.self_healing_orchestrator.calculate_ks_test", return_value=(0.1, 0.001)):
            anomalies = orchestrator._analyze_drift(shifted_metrics)
            assert len(anomalies) > 0
            assert anomalies[0]["type"] == "distribution_drift"

@pytest.mark.asyncio
async def test_orchestrator_start_stop(orchestrator):
    mock_source = MagicMock()
    mock_source.get_latest_metrics_async = AsyncMock(return_value=pd.DataFrame({"a": [1]}))
    
    # Use a small check interval
    orchestrator.check_interval = 0.1
    
    # Run in a task and stop it quickly
    task = asyncio.create_task(orchestrator.start(mock_source))
    await asyncio.sleep(0.2)
    orchestrator.stop()
    await task
    
    assert mock_source.get_latest_metrics_async.called
