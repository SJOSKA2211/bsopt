"""
Pickle Rick's Venv Enforcer
==========================
If you aren't in a venv, you aren't doing engineering.
"""

import os
import sys


def check_venv():
    # Detect if we are running in a virtual environment
    in_venv = (
        sys.prefix != sys.base_prefix or 
        os.environ.get("VIRTUAL_ENV") is not None or 
        os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    )
    
    if not in_venv:
        print("\n" + "="*60)
        print("ENVIRONMENT ERROR: VIRTUAL ENVIRONMENT NOT DETECTED")
        print("="*60)
        print("You are trying to run this in your global Python environment.")
        print("Please use a virtual environment to manage dependencies.")
        print("\nFIX IT:")
        print("  1. Create a venv: python3 -m venv .venv")
        print("  2. Activate it:   source .venv/bin/activate")
        print("  3. Install deps:  pip install -r requirements.txt")
        print("="*60 + "\n")
        sys.exit(1)

    # Check for required performance libraries
    import importlib.util
    for lib in ["msgspec", "numba"]:
        if importlib.util.find_spec(lib) is None:
            print(f"⚠️ WARNING: Performance library missing: {lib}.")
            print("Engine requires numba and msgspec for optimized pricing.")

if __name__ == "__main__":
    check_venv()