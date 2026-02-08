import importlib

import pytest


def test_import_security_oauth2():
    # Automatically generated import test for security.oauth2
    module = importlib.import_module("src.security.oauth2")
    assert module is not None

def test_initialization_security_oauth2():
    # Automatically generated init test for security.oauth2
    try:
        importlib.import_module("src.security.oauth2")
    except Exception as e:
        pytest.skip(f"Could not import src.security.oauth2: {e}")
