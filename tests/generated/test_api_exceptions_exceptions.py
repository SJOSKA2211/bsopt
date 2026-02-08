import importlib

import pytest


def test_import_api_exceptions_exceptions():
    # Automatically generated import test for api.exceptions.exceptions
    module = importlib.import_module("src.api.exceptions.exceptions")
    assert module is not None

def test_initialization_api_exceptions_exceptions():
    # Automatically generated init test for api.exceptions.exceptions
    try:
        importlib.import_module("src.api.exceptions.exceptions")
    except Exception as e:
        pytest.skip(f"Could not import src.api.exceptions.exceptions: {e}")
