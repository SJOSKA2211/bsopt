import traceback

try:
    import numpy as np

    from core.shared.math_utils import _vec_price_impl

    S = np.array([100.0], dtype=np.float64)
    K = np.array([100.0], dtype=np.float64)
    T = np.array([1.0], dtype=np.float64)
    sigma = np.array([0.2], dtype=np.float64)
    r = np.array([0.05], dtype=np.float64)
    q = np.array([0.02], dtype=np.float64)
    is_call = np.array([True], dtype=np.bool_)

    _vec_price_impl(S, K, T, sigma, r, q, is_call)
except Exception:
    traceback.print_exc()
