import numpy as np
import pytest

from src.shared.math_utils import calculate_price_core

def test_minimal_jit_scalar():
    res = calculate_price_core(100.0, 100.0, 1.0, 0.2, 0.05, 0.02, True)
    assert np.isclose(res, 9.22699372262055)

if __name__ == "__main__":
    pytest.main([__file__])
