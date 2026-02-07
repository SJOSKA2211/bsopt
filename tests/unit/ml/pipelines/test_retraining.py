from unittest.mock import AsyncMock, patch

import pytest

from src.ml.pipelines.retraining import NeuralGreeksRetrainer


@pytest.mark.asyncio
async def test_retrainer_trigger_success():
    # Mock MLOrchestrator
    with patch("src.ml.pipelines.retraining.MLOrchestrator") as mock_orch:
        mock_orch.return_value.run_training_pipeline = AsyncMock(
            return_value={"run_id": "test_run", "r2": 0.99}
        )

        retrainer = NeuralGreeksRetrainer()
        result = await retrainer.retrain_now()

        assert result["run_id"] == "test_run"
        mock_orch.return_value.run_training_pipeline.assert_called_with(
            model_type="nn", promote_to_production=True, n_samples=10000
        )


@pytest.mark.asyncio
async def test_retrainer_with_custom_samples():
    with patch("src.ml.pipelines.retraining.MLOrchestrator") as mock_orch:
        mock_orch.return_value.run_training_pipeline = AsyncMock(
            return_value={"run_id": "test_run"}
        )

        retrainer = NeuralGreeksRetrainer(n_samples=50000)
        await retrainer.retrain_now()

        mock_orch.return_value.run_training_pipeline.assert_called_with(
            model_type="nn", promote_to_production=True, n_samples=50000
        )


@pytest.mark.asyncio
async def test_retrainer_failure():
    with patch("src.ml.pipelines.retraining.MLOrchestrator") as mock_orch:
        mock_orch.return_value.run_training_pipeline = AsyncMock(
            side_effect=Exception("Retrain fail")
        )

        retrainer = NeuralGreeksRetrainer()
        with pytest.raises(Exception):
            await retrainer.retrain_now()
