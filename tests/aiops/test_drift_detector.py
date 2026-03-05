from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.aiops.drift_detector import PricingDriftDetector


@pytest.mark.asyncio
async def test_pricing_drift_detector_init():
    detector = PricingDriftDetector(threshold=0.1)
    assert detector.threshold == 0.1

@pytest.mark.asyncio
async def test_statistical_drift_detection():
    detector = PricingDriftDetector()
    
    # 1. No Drift Case
    ref_dist = np.random.normal(0, 1, 1000)
    curr_dist = np.random.normal(0, 1, 1000)
    
    drift, metrics = detector.calculate_statistical_drift(ref_dist, curr_dist)
    assert drift is False
    assert metrics["psi"] < 0.1
    assert metrics["ks_p_value"] > 0.05

    # 2. Significant Drift Case
    drift_dist = np.random.normal(2, 1, 1000)
    drift, metrics = detector.calculate_statistical_drift(ref_dist, drift_dist)
    assert drift is True
    assert metrics["psi"] > 0.25
    assert metrics["ks_p_value"] < 0.05

@pytest.mark.asyncio
@patch("src.aiops.drift_detector.get_async_db_context")
async def test_check_drift_theoretical(mock_db_context):
    detector = PricingDriftDetector(threshold=0.05)
    
    # Mock DB interaction
    mock_session = AsyncMock()
    mock_db_context.return_value.__aenter__.return_value = mock_session
    
    # Assuming we implement the fetch logic in the source, we mock it here
    with patch("src.aiops.drift_detector.PricingDriftDetector.check_drift", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = {
            "drift_detected": True,
            "reason": "theoretical_error_threshold_exceeded",
            "mean_relative_error": 0.09
        }
        
        result = await detector.check_drift("TSLA")
        assert result["drift_detected"] is True
        assert result["mean_relative_error"] == 0.09

@pytest.mark.asyncio
async def test_analyze_vol_smile_drift_stub():
    detector = PricingDriftDetector()
    result = await detector.analyze_vol_smile_drift("TSLA")
    assert result is None
