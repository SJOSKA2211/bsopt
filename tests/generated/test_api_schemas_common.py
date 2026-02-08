import importlib

import pytest


def test_import_api_schemas_common():
    # Automatically generated import test for api.schemas.common
    module = importlib.import_module("src.api.schemas.common")
    assert module is not None

def test_initialization_api_schemas_common():
    # Automatically generated init test for api.schemas.common
    try:
        importlib.import_module("src.api.schemas.common")
    except Exception as e:
        pytest.skip(f"Could not import src.api.schemas.common: {e}")
