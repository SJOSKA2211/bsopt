import sys
import os
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Path: {sys.path}")
try:
    import importlib.metadata
    try:
        version = importlib.metadata.version("strawberry-graphql")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    print(f"Strawberry version: {version}")
except ImportError:
    print("Strawberry not found")
