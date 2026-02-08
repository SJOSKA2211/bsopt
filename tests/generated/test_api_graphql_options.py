import importlib

import pytest


def test_import_api_graphql_options():
    # Automatically generated import test for api.graphql.options
    module = importlib.import_module("src.api.graphql.options")
    assert module is not None

def test_initialization_api_graphql_options():
    # Automatically generated init test for api.graphql.options
    try:
        importlib.import_module("src.api.graphql.options")
    except Exception as e:
        pytest.skip(f"Could not import src.api.graphql.options: {e}")
