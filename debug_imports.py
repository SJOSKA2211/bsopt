import sys
print("Step 1: sys imported")
try:
    from unittest.mock import AsyncMock, MagicMock
    print("Step 2: unittest.mock imported")
except Exception as e:
    print(f"Step 2 Failed: {e}")

try:
    import importlib.util
    print("Step 3: importlib.util imported")
except Exception as e:
    print(f"Step 3 Failed: {e}")

try:
    import importlib.machinery
    print("Step 4: importlib.machinery imported")
except Exception as e:
    print(f"Step 4 Failed: {e}")

print("Debug script finished")
