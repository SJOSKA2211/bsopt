import os
import sys

import ray
import structlog

# Project root
sys.path.insert(0, os.getcwd())

from src.shared.observability import setup_logging
from src.workers.math_worker import recalibrate_symbol

setup_logging()
logger = structlog.get_logger()

def smoke_test_ray():
    """Verify end-to-end Ray integration via math worker."""
    print("🧠 Starting Ray Smoke Test...")
    
    # 1. Ensure Ray is available
    if not ray.is_initialized():
        print("⏳ Initializing Ray connection (Address: auto)...")
        try:
            ray.init(address="auto")
        except Exception as e:
            print(f"❌ Failed to connect to Ray head node: {e}")
            return False
    
    print(f"✅ Connected to Ray cluster: {ray.cluster_resources()}")

    # 2. Trigger calibration task
    # Note: This will attempt to use the 'EQTY' symbol which might not have real data,
    # but we want to see if the Ray actor is triggered and if it fails gracefully or succeeds.
    symbol = "TEST_SYMBOL"
    print(f" Triggering calibration for {symbol}...")
    
    try:
        # We call the task function directly. It will use get_math_swarm() which uses MathActor.remote()
        result = recalibrate_symbol(symbol)
        print(f"📊 Result: {result}")
        
        if result.get("status") == "success" or result.get("reason") == "no_data":
            print("✅ Ray task execution flow verified.")
            return True
        else:
            print(f"❌ Ray task failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Smoke test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = smoke_test_ray()
    if success:
        print(" Ray Integration: SUCCESS")
        sys.exit(0)
    else:
        print("Jerry-work detected in Ray integration.")
        sys.exit(1)
