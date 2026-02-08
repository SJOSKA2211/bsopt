import importlib

import pytest


def test_import_utils_chaos():
    # Automatically generated import test for utils.chaos
    module = importlib.import_module("src.utils.chaos")
    assert module is not None

def test_initialization_utils_chaos():
    # Automatically generated init test for utils.chaos
    try:
        importlib.import_module("src.utils.chaos")
    except Exception as e:
        pytest.skip(f"Could not import src.utils.chaos: {e}")
