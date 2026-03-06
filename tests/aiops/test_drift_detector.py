
import pytest

from src.aiops.drift_detector import PricingDriftDetector


@pytest.mark.asyncio
async def test_check_drift_theoretical():
    detector = PricingDriftDetector(threshold=0.05)

    import numpy as np
    current_data = np.array([0.1, 0.2, 0.3])
    reference_data = np.array([0.11, 0.19, 0.31])

    result = await detector.check_drift(
        symbol="BTC/USD", current_data=current_data, reference_data=reference_data
    )
    assert result["symbol"] == "BTC/USD"
    assert "drift_detected" in result
    assert isinstance(result["reasons"], list)


@pytest.mark.asyncio
async def test_drift_detector_initialization():
    detector = PricingDriftDetector(threshold=0.1)
    assert detector.threshold == 0.1
    assert detector.logger is not None
