import pytest

def test_performance_benchmarks_skipped():
    """
    Performance benchmarks verification is skipped due to user instruction
    not to run or build Docker containers.
    Manual execution of benchmarks is required to validate optimizations.
    """
    pytest.skip("Performance benchmarks explicitly skipped by user instruction.")
