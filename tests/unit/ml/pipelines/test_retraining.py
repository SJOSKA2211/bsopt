from unittest.mock import patch

import pytest

from src.ml.pipelines.retraining import NeuralGreeksRetrainer


@pytest.fixture
def retrainer():
    return NeuralGreeksRetrainer(n_samples=100)


@pytest.mark.asyncio
async def test_retrainer_trigger_success(retrainer):
    with patch("src.ml.pipelines.retraining.train_neural_network") as mock_train:
        mock_train.return_value = "/tmp/mock_model.pt"
        res = await retrainer.retrain_now()
        assert res["status"] == "success"
        assert res["model_path"] == "/tmp/mock_model.pt"


@pytest.mark.asyncio
async def test_retrainer_with_drift(retrainer):
    with patch("src.aiops.data_drift_detector.DataDriftDetector.detect_drift") as mock_drift:
        mock_drift.return_value = {"is_drift_detected": True}
        with pytest.raises(ValueError, match="data drift"):
            await retrainer.retrain_now(data=[1, 2, 3])


@pytest.mark.asyncio
async def test_retrainer_failure(retrainer):
    with patch("src.ml.pipelines.retraining.train_neural_network") as mock_train:
        mock_train.side_effect = Exception("Training failed")
        with pytest.raises(Exception, match="Training failed"):
            await retrainer.retrain_now()
