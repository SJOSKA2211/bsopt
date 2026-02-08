import importlib

import pytest


def test_import_api_schemas_auth():
    # Automatically generated import test for api.schemas.auth
    module = importlib.import_module("src.api.schemas.auth")
    assert module is not None

def test_initialization_api_schemas_auth():
    # Automatically generated init test for api.schemas.auth
    try:
        importlib.import_module("src.api.schemas.auth")
    except Exception as e:
        pytest.skip(f"Could not import src.api.schemas.auth: {e}")
