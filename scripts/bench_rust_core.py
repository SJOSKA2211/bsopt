import time

import Manifold_core as mc
import numpy as np


def bench():
    n = 10_000_000
    S = np.full(n, 100.0, dtype=np.float64)
    K = np.full(n, 100.0, dtype=np.float64)
    T = np.full(n, 1.0, dtype=np.float64)
    sigma = np.full(n, 0.2, dtype=np.float64)
    r = np.full(n, 0.05, dtype=np.float64)
    q = np.full(n, 0.02, dtype=np.float64)
    is_call = np.full(n, True, dtype=bool)
    
    # Warmup
    mc.batch_black_scholes(S[:1000], K[:1000], T[:1000], sigma[:1000], r[:1000], q[:1000], is_call[:1000])
    
    start = time.time()
    res = mc.batch_black_scholes(S, K, T, sigma, r, q, is_call)
    end = time.time()
    
    duration = end - start
    throughput = n / duration
    print(f"Local Rust Throughput (n={n}): {throughput:,.2f} sims/sec")
    print(f"Time: {duration:.4f}s")

if __name__ == "__main__":
    bench()
