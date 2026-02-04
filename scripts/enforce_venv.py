import sys
import os

def check_venv():
    """
    Strictly enforce that the script is running inside a virtual environment.
    """
    # Check for VIRTUAL_ENV environment variable or sys.base_prefix check
    in_venv = (sys.prefix != sys.base_prefix) or (hasattr(sys, "real_prefix")) or (os.environ.get("VIRTUAL_ENV") is not None)
    
    if not in_venv:
        print("❌ CRITICAL ERROR: Not running in a virtual environment!")
        print("   You are attempting to run BSOpt in the global Python environment.")
        print("   This is strictly forbidden by the Pickle Rick protocol.")
        print("\n   PLEASE RUN:")
        print("   source .venv/bin/activate")
        print("   (or create one: python3 -m venv .venv)")
        sys.exit(1)

if __name__ == "__main__":
    check_venv()
