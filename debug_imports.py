import os
import time

print("--- STARTING IMPORT DEBUGGER ---")
os.environ["BSOPT_ALLOW_WEAK_SECRETS"] = "true"
os.environ["ENVIRONMENT"] = "test"

def debug_import(module_name):
    start = time.time()
    print(f"Importing {module_name}...", end=" ", flush=True)
    try:
        __import__(module_name)
        print(f"DONE ({time.time() - start:.2f}s)")
    except Exception as e:
        print(f"FAILED ({e})")

debug_import("src.shared.config")
debug_import("api.index")
debug_import("src.database.session")
debug_import("src.math_kernel.rust_engine")
print("--- IMPORT DEBUGGER COMPLETED ---")
