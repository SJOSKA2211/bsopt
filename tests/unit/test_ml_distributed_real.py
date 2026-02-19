import pytest
import ray

from src.ml.distributed_training import HAS_RAY_TRAIN, BSOptDistributedTrainer


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for testing."""
    if HAS_RAY_TRAIN:
        ray.init(ignore_reinit_error=True)
        yield
        ray.shutdown()
    else:
        pytest.skip("Ray Train not available")


def test_distributed_trainer_execution(ray_init):
    """Verify that BSOptDistributedTrainer executes without error."""
    trainer = BSOptDistributedTrainer(num_workers=1, use_gpu=False)

    config = {
        "lr": 1e-3,
        "epochs": 1,
        "dataset_size": 64,
        "batch_size": 16,
        "experiment_name": "Test_Distributed_Run",
    }

    result = trainer.run(config)

    assert result is not None
    # Ray Train results have a .metrics attribute
    assert hasattr(result, "metrics")
    # V2 Trainer should have completed
    print(f"Distributed Test Metrics: {result.metrics}")
