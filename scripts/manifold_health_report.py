from src.math_kernel.rust_engine import get_rust_metrics
from src.math_kernel.factory import PricingEngineFactory
from src.math_kernel.models import BSParameters
import numpy as np

def run_health_report():
    print("=" * 60)
    print("Manifold Core Health & Performance Report")
    print("=" * 60)
    
    # Run some pricing work to populate metrics
    rust_engine = PricingEngineFactory.get_engine("rust")
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.0
    )
    
    print("[*] Warming up Rust Engine...")
    for _ in range(100):
        rust_engine.price_european(params)
        
    print("[*] Running Batch Computations...")
    # Batch pricing if supported by engine wrapper
    S = np.random.uniform(90, 110, 1000)
    K = np.random.uniform(90, 110, 1000)
    T = np.ones(1000)
    V = np.full(1000, 0.2)
    R = np.full(1000, 0.05)
    Q = np.zeros(1000)
    IsCall = np.ones(1000, dtype=bool)
    
    from src.math_kernel.rust_engine import price_black_scholes, RustTickBuffer as TickDataBuffer
    import os
    
    tick_file = "/tmp/report_ticks.bin"
    with open(tick_file, "wb") as f:
        f.write(os.urandom(1024 * 32))
        
    buffer = TickDataBuffer(tick_file)
    print(f"[*] Active MMap Buffers (internal check): {buffer.size} bytes mapped")
    
    price_black_scholes(S, K, T, V, R, Q, IsCall)
    
    print("\n[+] Health Metrics (Prometheus Format):")
    metrics = get_rust_metrics()
    print(metrics)
    
    if "manifold_thread_pool_size" in metrics:
        print("[✔] Rayon thread pool metrics detected")
    if "manifold_mmap_active_buffers" in metrics:
        print("[✔] MMap buffer metrics detected")
        
    os.remove(tick_file)
    print("=" * 60)

if __name__ == "__main__":
    run_health_report()
