import os

import pytest
import ray

from src.ml.reinforcement_learning.train import train_distributed


@pytest.fixture(scope="module")
def ray_cluster():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_train_distributed_success(ray_cluster):
    """Verify that train_distributed launches multiple instances and returns results."""
    # We'll use a very small number of timesteps for testing
    num_instances = 2
    timesteps = 100

    # Run distributed training
    # ray_address=None for local testing
    results, best_result = train_distributed(
        num_instances=num_instances, total_timesteps=timesteps, ray_address=None
    )

    assert results is not None
    assert len(results) == num_instances
    assert "mean_reward" in best_result
    assert "model_path" in best_result

    # Each result should be a dict with metadata
    for res in results:
        assert isinstance(res, dict)
        assert "run_id" in res
        assert "model_path" in res
        assert os.path.exists(res["model_path"])


def test_distributed_trainer_selects_best(ray_cluster):
    """Verify that the distributed trainer correctly identifies the best model."""
    num_instances = 2
    timesteps = 100

    results, best_result = train_distributed(
        num_instances=num_instances, total_timesteps=timesteps, ray_address=None
    )

    # Check if best_result actually has the maximum reward
    max_reward = max(res["mean_reward"] for res in results)
    assert best_result["mean_reward"] == max_reward
