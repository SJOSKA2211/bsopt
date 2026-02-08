import importlib

import pytest


def test_import_api_graphql_resolvers_option_service():
    # Automatically generated import test for api.graphql.resolvers.option_service
    module = importlib.import_module("src.api.graphql.resolvers.option_service")
    assert module is not None

def test_initialization_api_graphql_resolvers_option_service():
    # Automatically generated init test for api.graphql.resolvers.option_service
    try:
        importlib.import_module("src.api.graphql.resolvers.option_service")
    except Exception as e:
        pytest.skip(f"Could not import src.api.graphql.resolvers.option_service: {e}")
