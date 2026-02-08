import importlib

import pytest


def test_import_utils_lazy_import():
    # Automatically generated import test for utils.lazy_import
    module = importlib.import_module("src.utils.lazy_import")
    assert module is not None

def test_initialization_utils_lazy_import():
    # Automatically generated init test for utils.lazy_import
    try:
        importlib.import_module("src.utils.lazy_import")
    except Exception as e:
        pytest.skip(f"Could not import src.utils.lazy_import: {e}")
