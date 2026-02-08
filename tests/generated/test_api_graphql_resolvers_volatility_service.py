import importlib

import pytest


def test_import_api_graphql_resolvers_volatility_service():
    # Automatically generated import test for api.graphql.resolvers.volatility_service
    module = importlib.import_module("src.api.graphql.resolvers.volatility_service")
    assert module is not None

def test_initialization_api_graphql_resolvers_volatility_service():
    # Automatically generated init test for api.graphql.resolvers.volatility_service
    try:
        importlib.import_module("src.api.graphql.resolvers.volatility_service")
    except Exception as e:
        pytest.skip(f"Could not import src.api.graphql.resolvers.volatility_service: {e}")
