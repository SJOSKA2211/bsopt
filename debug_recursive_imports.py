import sys
import os
import pkgutil
import importlib
import time

print("Starting import_debug.py")
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('.'))

def walk_packages(path=None, prefix=""):
    for _, name, ispkg in pkgutil.iter_modules(path, prefix):
        print(f"DTO: Importing {name}...")
        start = time.time()
        try:
            importlib.import_module(name)
            print(f"SUCCESS: {name} ({time.time() - start:.4f}s)")
        except Exception as e:
            print(f"FAILED: {name} - {e}")
        
        if ispkg:
            # Recurse? pkgutil.iter_modules doesn't recurse automatically
            pass

print("Scanning src...")
walk_packages(['src'], 'src.')
print("Scan complete")
