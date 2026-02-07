import ray
import structlog

from src.ml.reinforcement_learning.train import train_distributed

logger = structlog.get_logger()


def verify_ray_scaling():
    """
    Verify that we can start a local Ray instance and run parallel training.
    """
    logger.info("verifying_ray_scaling_start")

    try:
        # Start a local Ray instance with 2 workers
        ray.init(num_cpus=2, ignore_reinit_error=True)

        # Run 2 parallel instances of 100 timesteps each
        results = train_distributed(
            num_instances=2, total_timesteps=100, ray_address=None
        )

        if results and len(results) == 2:
            logger.info("verifying_ray_scaling_success")
            return True
        else:
            logger.error(
                "verifying_ray_scaling_failed", message="Unexpected results from Ray"
            )
            return False

    except Exception as e:
        logger.error("verifying_ray_scaling_error", error=str(e))
        return False
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    success = verify_ray_scaling()
    if success:
        print("Ray Scaling Verification: PASSED")
        exit(0)
    else:
        print("Ray Scaling Verification: FAILED")
        exit(1)
