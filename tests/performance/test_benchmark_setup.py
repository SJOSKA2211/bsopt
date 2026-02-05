import time


def test_benchmark_fixture(benchmark):
    """Test that pytest-benchmark is installed and working."""
    def expensive_operation():
        time.sleep(0.001)
        return True
        
    result = benchmark(expensive_operation)
    assert result is True
