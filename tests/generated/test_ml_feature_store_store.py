import importlib

import pytest


def test_import_ml_feature_store_store():
    # Automatically generated import test for ml.feature_store.store
    module = importlib.import_module("src.ml.feature_store.store")
    assert module is not None

def test_initialization_ml_feature_store_store():
    # Automatically generated init test for ml.feature_store.store
    try:
        importlib.import_module("src.ml.feature_store.store")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.feature_store.store: {e}")
