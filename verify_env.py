import sys
import os
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Path: {sys.path}")
try:
    import strawberry
    print(f"Strawberry version: {strawberry.__version__}")
except ImportError:
    print("Strawberry not found")
