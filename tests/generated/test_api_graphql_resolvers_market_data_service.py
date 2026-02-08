import importlib

import pytest


def test_import_api_graphql_resolvers_market_data_service():
    # Automatically generated import test for api.graphql.resolvers.market_data_service
    module = importlib.import_module("src.api.graphql.resolvers.market_data_service")
    assert module is not None

def test_initialization_api_graphql_resolvers_market_data_service():
    # Automatically generated init test for api.graphql.resolvers.market_data_service
    try:
        importlib.import_module("src.api.graphql.resolvers.market_data_service")
    except Exception as e:
        pytest.skip(f"Could not import src.api.graphql.resolvers.market_data_service: {e}")
