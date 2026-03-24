"""
Performance Library Checker
==========================
Checks for required high-performance modules.
"""

import importlib.util

def check_performance_libs():
    libs = ["msgspec", "numba"]
    for lib in libs:
        if importlib.util.find_spec(lib) is None:
            print(f"⚠️ WARNING: Performance library missing: {lib}.")
            print("Engine requires numba and msgspec for optimized pricing.")

if __name__ == "__main__":
    check_performance_libs()
