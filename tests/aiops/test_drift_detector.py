from unittest.mock import AsyncMock, patch

import pytest

from src.aiops.drift_detector import PricingDriftDetector


@pytest.mark.asyncio
@patch("src.aiops.drift_detector.get_async_db_context")
async def test_check_drift_theoretical(mock_db_context):
    detector = PricingDriftDetector(threshold=0.05)

    # Mock DB interaction
    mock_session = AsyncMock()
    mock_db_context.return_value.__aenter__.return_value = mock_session

    # Assuming we implement the fetch logic in the source, we mock it here
    with patch(
        "src.aiops.drift_detector.PricingDriftDetector.check_drift", new_callable=AsyncMock
    ) as mock_check:
        mock_check.return_value = {
            "drift_detected": True,
            "reason": "theoretical_error_threshold_exceeded",
            "mean_relative_error": 0.09,
        }

        result = await detector.check_drift()
        assert result["drift_detected"] is True
        assert result["mean_relative_error"] > 0.05


@pytest.mark.asyncio
async def test_drift_detector_initialization():
    detector = PricingDriftDetector(threshold=0.1)
    assert detector.threshold == 0.1
    assert detector.logger is not None
