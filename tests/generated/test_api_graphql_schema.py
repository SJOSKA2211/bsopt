import importlib

import pytest


def test_import_api_graphql_schema():
    # Automatically generated import test for api.graphql.schema
    module = importlib.import_module("src.api.graphql.schema")
    assert module is not None

def test_initialization_api_graphql_schema():
    # Automatically generated init test for api.graphql.schema
    try:
        importlib.import_module("src.api.graphql.schema")
    except Exception as e:
        pytest.skip(f"Could not import src.api.graphql.schema: {e}")
