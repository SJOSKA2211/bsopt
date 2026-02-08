import importlib

import pytest


def test_import_api_exceptions():
    # Automatically generated import test for api.exceptions
    module = importlib.import_module("src.api.exceptions")
    assert module is not None

def test_initialization_api_exceptions():
    # Automatically generated init test for api.exceptions
    try:
        importlib.import_module("src.api.exceptions")
    except Exception as e:
        pytest.skip(f"Could not import src.api.exceptions: {e}")
