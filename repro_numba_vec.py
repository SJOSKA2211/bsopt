import sys
import os
import numpy as np
sys.path.append('.')

from src.shared.math_utils import calculate_price, _vec_price_impl
from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.models import BSParameters

# Force 1D arrays to trigger the vectorized path
S = np.array([100.0], dtype=np.float64)
K = np.array([100.0], dtype=np.float64)
T = np.array([1.0], dtype=np.float64)
sigma = np.array([0.2], dtype=np.float64)
r = np.array([0.05], dtype=np.float64)
q = np.array([0.02], dtype=np.float64)
is_call = np.array([True], dtype=np.bool_)

print("Starting vectorized pricing test...")
try:
    res = _vec_price_impl(S, K, T, sigma, r, q, is_call)
    print(f"Result: {res}")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
