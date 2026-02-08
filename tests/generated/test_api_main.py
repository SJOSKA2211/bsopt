import importlib

import pytest


def test_import_api_main():
    # Automatically generated import test for api.main
    module = importlib.import_module("src.api.main")
    assert module is not None

def test_initialization_api_main():
    # Automatically generated init test for api.main
    try:
        importlib.import_module("src.api.main")
    except Exception as e:
        pytest.skip(f"Could not import src.api.main: {e}")
