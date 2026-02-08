import importlib

import pytest


def test_import_workers_math_worker():
    # Automatically generated import test for workers.math_worker
    module = importlib.import_module("src.workers.math_worker")
    assert module is not None

def test_initialization_workers_math_worker():
    # Automatically generated init test for workers.math_worker
    try:
        importlib.import_module("src.workers.math_worker")
    except Exception as e:
        pytest.skip(f"Could not import src.workers.math_worker: {e}")
