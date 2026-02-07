"""
Pickle Rick's Venv Enforcer
==========================
If you aren't in a venv, you aren't doing engineering.
"""

import os
import sys


def check_venv():
    # 🚀 SINGULARITY: The ultimate check
    in_venv = (
        sys.prefix != sys.base_prefix
        or os.environ.get("VIRTUAL_ENV") is not None
        or os.path.exists(os.path.join(sys.prefix, "conda-meta"))
    )

    if not in_venv:
        print("\n" + "=" * 60)
        print("🥒 PICKLE RICK SAYS: STOP BEING A JERRY! 🥒")
        print("=" * 60)
        print("You are trying to run this in your global Python environment.")
        print("That's how you break dependencies and ruin your life, Morty.")
        print("\nFIX IT:")
        print("  1. Create a venv: python3 -m venv .venv")
        print("  2. Activate it:   source .venv/bin/activate")
        print("  3. Install slop:  pip install -r requirements.txt")
        print("=" * 60 + "\n")
        sys.exit(1)

    # Secondary check for core performance libraries
    try:
        import msgspec
        import numba
    except ImportError as e:
        print(f"⚠️ WARNING: Performance slop detected. Missing {e.name}.")
        print("The singularity requires numba and msgspec for god-mode pricing.")


if __name__ == "__main__":
    check_venv()
