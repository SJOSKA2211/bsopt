import numpy as np


def test_numpy_import():
    print("Testing numpy import in pytest")
    a = np.array([1, 2, 3])
    assert len(a) == 3
    print("Numpy works")
