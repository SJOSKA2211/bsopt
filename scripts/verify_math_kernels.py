import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('.'))

try:
    from src.shared.math_utils import calculate_price, calculate_greeks
    print("[+] Successfully imported math_utils")
    
    # Test parameters
    s = np.array([100.0, 110.0])
    k = np.array([100.0, 100.0])
    t = np.array([1.0, 0.5])
    v = np.array([0.2, 0.3])
    r = np.array([0.05, 0.05])
    q = np.array([0.0, 0.0])
    is_call = np.array([True, False])
    
    print("[*] Testing vectorized pricing...")
    prices = calculate_price(s, k, t, v, r, q, is_call)
    print(f"[+] Prices: {prices}")
    
    print("[*] Testing vectorized greeks...")
    d, g, th, v_g, rh = calculate_greeks(s, k, t, v, r, q, is_call)
    print(f"[+] Delta: {d}")
    print(f"[+] Gamma: {g}")
    print(f"[+] Theta: {th}")
    
    print("\n[SUCCESS] Math kernels are operational and vectorized.")
    
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)
