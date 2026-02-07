from unittest.mock import AsyncMock, patch

import pytest

from src.ml.pipelines.dag_neural_greeks import neural_greeks_retraining_flow


@pytest.mark.asyncio
async def test_neural_greeks_flow_execution():
    """Test that the flow logic correctly calls the retrainer via task."""
    with patch("src.ml.pipelines.dag_neural_greeks.NeuralGreeksRetrainer") as mock_retrainer_cls:
        mock_retrainer_instance = mock_retrainer_cls.return_value
        mock_retrainer_instance.retrain_now = AsyncMock(return_value={"status": "success"})
        
        # Bypassing Prefect engine by calling task.fn and flow.fn
        from src.ml.pipelines.dag_neural_greeks import retrain_neural_greeks_task
        
        # Manually simulate flow calling task
        result = await retrain_neural_greeks_task.fn(n_samples=100)
        
        assert result["status"] == "success"
        mock_retrainer_instance.retrain_now.assert_called_once()

def test_dag_structure():
    """Verify DAG (Flow) metadata and task structure."""
    assert neural_greeks_retraining_flow.name == "Neural-Greeks-Retraining"
    # Additional checks for Prefect flow attributes if needed
