import sys
import os
import time

print("Starting debug_pricing.py")
try:
    print("Importing numpy...")
    import numpy as np
    print("Numpy imported.")
    
    print("Importing BlackScholesEngine...")
    # Add src to path
    sys.path.insert(0, os.path.abspath('.'))
    sys.path.insert(0, os.path.abspath('src'))
    
    from src.pricing.black_scholes import BlackScholesEngine
    print("BlackScholesEngine imported.")

    print("Running pricing...")
    start = time.time()
    price = BlackScholesEngine.price_options(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call")
    end = time.time()
    print(f"Price: {price}")
    print(f"Time: {end - start:.4f}s")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Finished debug_pricing.py")
