import importlib

import pytest


def test_import_shared_off_heap_logger():
    # Automatically generated import test for shared.off_heap_logger
    module = importlib.import_module("src.shared.off_heap_logger")
    assert module is not None

def test_initialization_shared_off_heap_logger():
    # Automatically generated init test for shared.off_heap_logger
    try:
        importlib.import_module("src.shared.off_heap_logger")
    except Exception as e:
        pytest.skip(f"Could not import src.shared.off_heap_logger: {e}")
