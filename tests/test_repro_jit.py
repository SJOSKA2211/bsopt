import numpy as np
import pytest

from src.shared.math_utils import _vec_price_impl


def test_minimal_jit_vec():
    import os

    from src.shared.math_utils import calculate_price_core

    print(f"\nNUMBA_DISABLE_JIT: {os.getenv('NUMBA_DISABLE_JIT')}")
    print(f"type(_vec_price_impl): {type(_vec_price_impl)}")
    print(f"type(calculate_price_core): {type(calculate_price_core)}")
    S = np.array([100.0], dtype=np.float64)
    K = np.array([100.0], dtype=np.float64)
    T = np.array([1.0], dtype=np.float64)
    sigma = np.array([0.2], dtype=np.float64)
    r = np.array([0.05], dtype=np.float64)
    q = np.array([0.02], dtype=np.float64)
    is_call = np.array([True], dtype=np.bool_)

    res = _vec_price_impl(S, K, T, sigma, r, q, is_call)
    assert np.isclose(res[0], 9.22699372262055)


if __name__ == "__main__":
    pytest.main([__file__])