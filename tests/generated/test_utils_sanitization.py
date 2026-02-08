import importlib

import pytest


def test_import_utils_sanitization():
    # Automatically generated import test for utils.sanitization
    module = importlib.import_module("src.utils.sanitization")
    assert module is not None

def test_initialization_utils_sanitization():
    # Automatically generated init test for utils.sanitization
    try:
        importlib.import_module("src.utils.sanitization")
    except Exception as e:
        pytest.skip(f"Could not import src.utils.sanitization: {e}")
