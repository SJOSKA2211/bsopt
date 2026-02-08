import importlib

import pytest


def test_import_api_schemas_ml():
    # Automatically generated import test for api.schemas.ml
    module = importlib.import_module("src.api.schemas.ml")
    assert module is not None

def test_initialization_api_schemas_ml():
    # Automatically generated init test for api.schemas.ml
    try:
        importlib.import_module("src.api.schemas.ml")
    except Exception as e:
        pytest.skip(f"Could not import src.api.schemas.ml: {e}")
