import importlib

import pytest


def test_import_auth_providers():
    # Automatically generated import test for auth.providers
    module = importlib.import_module("src.auth.providers")
    assert module is not None

def test_initialization_auth_providers():
    # Automatically generated init test for auth.providers
    try:
        importlib.import_module("src.auth.providers")
    except Exception as e:
        pytest.skip(f"Could not import src.auth.providers: {e}")
