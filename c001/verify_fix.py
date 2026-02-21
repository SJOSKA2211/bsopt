import sys
import os

# Add src to sys.path
sys.path.append(os.getcwd())

print("Testing API import...")
try:
    from src.api.main import app
    print("✅ API import successful")
except Exception as e:
    print(f"❌ API import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting Quant Utils import and functions...")
try:
    from src.pricing.quant_utils import scalar_bs_price_jit, gpu_mc_european_price
    print("✅ Quant Utils import successful")
    
    # Test scalar_bs_price_jit
    p = scalar_bs_price_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True)
    print(f"✅ scalar_bs_price_jit result: {p}")
    
    # Test gpu_mc_european_price (CPU fallback)
    p_mc, err = gpu_mc_european_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 1000, True, True)
    print(f"✅ gpu_mc_european_price result: {p_mc} +/- {err}")
    
except Exception as e:
    print(f"❌ Quant Utils test failed: {e}")
    import traceback
    traceback.print_exc()
