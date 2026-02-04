import pytest
import numpy as np
from src.pricing.quant_utils import batch_bs_price_jit, batch_greeks_jit
from src.pricing.models import BSParameters

@pytest.fixture
def benchmark_data():
    """Generate standardized data for pricing benchmarks."""
    n = 10000
    S = np.random.uniform(90, 110, n).astype(np.float32)
    K = np.random.uniform(90, 110, n).astype(np.float32)
    T = np.random.uniform(0.1, 2.0, n).astype(np.float32)
    sigma = np.random.uniform(0.1, 0.5, n).astype(np.float32)
    r = np.full(n, 0.05, dtype=np.float32)
    q = np.full(n, 0.02, dtype=np.float32)
    is_call = np.random.choice([True, False], n)
    return S, K, T, sigma, r, q, is_call

def test_benchmark_bs_price_jit(benchmark, benchmark_data):
    """
    Benchmarks the Numba JIT Black-Scholes pricing kernel.
    Ensures that any changes to quant_utils.py are statistically verified.
    """
    S, K, T, sigma, r, q, is_call = benchmark_data
    
    # Run the benchmark
    # The 'benchmark' fixture is provided by pytest-benchmark
    result = benchmark(batch_bs_price_jit, S, K, T, sigma, r, q, is_call)
    
    assert len(result) == len(S)
    assert result.dtype == np.float32

def test_benchmark_greeks_jit(benchmark, benchmark_data):
    """Benchmarks the Numba JIT Greeks kernel."""
    S, K, T, sigma, r, q, is_call = benchmark_data
    
    # benchmark() handles timing, warm-up, and statistical analysis
    delta, gamma, vega, theta, rho = benchmark(
        batch_greeks_jit, S, K, T, sigma, r, q, is_call
    )
    
    assert len(delta) == len(S)
    assert delta.dtype == np.float32
