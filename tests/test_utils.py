from typing import Any

import numpy as np


def assert_equal(
    actual: Any, expected: Any, tolerance: float = 1e-7, message: str = ""
):
    """
    Custom assertion helper to compare values with tolerance for floats
    and support for numpy arrays.
    """
    if isinstance(actual, (float, np.float64, np.float32)) and isinstance(
        expected, (float, int, np.float64, np.float32)
    ):
        if not abs(actual - expected) < tolerance:
            raise AssertionError(
                f"{message}: Expected {expected}, but got {actual} (diff={abs(actual-expected)})"
                if message
                else f"Expected {expected}, but got {actual}"
            )
    elif isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_allclose(actual, expected, atol=tolerance, err_msg=message)
    else:
        if not actual == expected:
            raise AssertionError(
                f"{message}: Expected {expected}, but got {actual}"
                if message
                else f"Expected {expected}, but got {actual}"
            )
