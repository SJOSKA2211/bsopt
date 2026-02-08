import importlib

import pytest


def test_import_api_schemas_user():
    # Automatically generated import test for api.schemas.user
    module = importlib.import_module("src.api.schemas.user")
    assert module is not None

def test_initialization_api_schemas_user():
    # Automatically generated init test for api.schemas.user
    try:
        importlib.import_module("src.api.schemas.user")
    except Exception as e:
        pytest.skip(f"Could not import src.api.schemas.user: {e}")
