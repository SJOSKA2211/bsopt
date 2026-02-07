import asyncio
from typing import Any

from prefect import flow, task

from src.ml.pipelines.retraining import NeuralGreeksRetrainer


@task(name="Retrain-Neural-Greeks")
async def retrain_neural_greeks_task(n_samples: int) -> dict[str, Any]:
    """Prefect task to trigger NeuralGreeksRetrainer."""
    retrainer = NeuralGreeksRetrainer(n_samples=n_samples)
    return await retrainer.retrain_now()

@flow(name="Neural-Greeks-Retraining")
async def neural_greeks_retraining_flow(n_samples: int = 10000):
    """
    Weekly Retraining DAG for Neural Greeks Engine.
    Triggered by schedule or manual dispatch.
    """
    return await retrain_neural_greeks_task(n_samples)

if __name__ == "__main__":
    # For local testing
    asyncio.run(neural_greeks_retraining_flow(n_samples=100))
