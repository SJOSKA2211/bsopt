def test_ray_import():
    """Test that ray is installed and can be imported."""
    import ray

    assert ray.__version__ is not None


def test_sb3_import():
    """Test that stable-baselines3 is installed and can be imported."""
    import stable_baselines3

    assert stable_baselines3.__version__ is not None


def test_gymnasium_import():
    """Test that gymnasium is installed and can be imported."""
    import gymnasium

    assert gymnasium.__version__ is not None


def test_ray_distributed_execution():
    """Test that Ray can execute a remote function and return results."""
    import ray

    if ray.is_initialized():
        ray.shutdown()

    # Initialize Ray locally with 2 CPUs
    ray.init(num_cpus=2, ignore_reinit_error=True)

    @ray.remote
    def remote_fn(x):
        return x * x

    # Execute remote tasks
    futures = [remote_fn.remote(i) for i in range(4)]
    results = ray.get(futures)

    assert results == [0, 1, 4, 9]

    ray.shutdown()
