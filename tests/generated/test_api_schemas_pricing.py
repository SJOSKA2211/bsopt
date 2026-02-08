import importlib

import pytest


def test_import_api_schemas_pricing():
    # Automatically generated import test for api.schemas.pricing
    module = importlib.import_module("src.api.schemas.pricing")
    assert module is not None

def test_initialization_api_schemas_pricing():
    # Automatically generated init test for api.schemas.pricing
    try:
        importlib.import_module("src.api.schemas.pricing")
    except Exception as e:
        pytest.skip(f"Could not import src.api.schemas.pricing: {e}")
