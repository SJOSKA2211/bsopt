import numpy as np

from src.shared.math_utils import calculate_price


def test_repro():
    S = np.atleast_1d(100.0).astype(np.float64)
    K = np.atleast_1d(100.0).astype(np.float64)
    T = np.atleast_1d(1.0).astype(np.float64)
    sigma = np.atleast_1d(0.2).astype(np.float64)
    r = np.atleast_1d(0.05).astype(np.float64)
    q = np.atleast_1d(0.02).astype(np.float64)
    is_call = np.atleast_1d(True).astype(bool)
    
    print(f"S shape: {S.shape}, is_call shape: {is_call.shape}")
    print("Testing calculate_price...")
    price = calculate_price(S, K, T, sigma, r, q, is_call)
    print(f"Price: {price}")

if __name__ == "__main__":
    try:
        test_repro()
    except Exception as e:
        print(f"Caught error: {e}")
        import traceback
        traceback.print_exc()
