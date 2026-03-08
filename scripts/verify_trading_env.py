import numpy as np
from src.ml.reinforcement_learning.trading_env import TradingEnvironment

def verify_env():
    print("Verifying TradingEnvironment with Numba JIT...")
    env = TradingEnvironment()
    obs, _ = env.reset()
    print(f"Initial Obs Shape: {obs.shape}")
    
    action = np.random.uniform(-1, 1, 10).astype(np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step Result:")
    print(f"  - Next Obs Shape: {next_obs.shape}")
    print(f"  - Reward: {reward:.6f}")
    print(f"  - Portfolio Value: {info['portfolio_value']:.2f}")
    
    # Run 1000 steps to warm up JIT and check performance
    import time
    start = time.perf_counter()
    for _ in range(1000):
        env.step(np.random.uniform(-1, 1, 10).astype(np.float32))
    duration = time.perf_counter() - start
    print(f"Performance: 1000 steps in {duration:.4f}s ({duration/1000*1e6:.2f} us/step)")

if __name__ == "__main__":
    try:
        verify_env()
        print("Status: Success")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
