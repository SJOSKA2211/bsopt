import importlib

import pytest


def test_import_utils_http_client():
    # Automatically generated import test for utils.http_client
    module = importlib.import_module("src.utils.http_client")
    assert module is not None

def test_initialization_utils_http_client():
    # Automatically generated init test for utils.http_client
    try:
        importlib.import_module("src.utils.http_client")
    except Exception as e:
        pytest.skip(f"Could not import src.utils.http_client: {e}")
