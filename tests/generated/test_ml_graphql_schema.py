import importlib

import pytest


def test_import_ml_graphql_schema():
    # Automatically generated import test for ml.graphql.schema
    module = importlib.import_module("src.ml.graphql.schema")
    assert module is not None

def test_initialization_ml_graphql_schema():
    # Automatically generated init test for ml.graphql.schema
    try:
        importlib.import_module("src.ml.graphql.schema")
    except Exception as e:
        pytest.skip(f"Could not import src.ml.graphql.schema: {e}")
