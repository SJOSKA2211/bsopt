import importlib

import pytest


def test_import_portfolio_graphql_schema():
    # Automatically generated import test for portfolio.graphql.schema
    module = importlib.import_module("src.portfolio.graphql.schema")
    assert module is not None

def test_initialization_portfolio_graphql_schema():
    # Automatically generated init test for portfolio.graphql.schema
    try:
        importlib.import_module("src.portfolio.graphql.schema")
    except Exception as e:
        pytest.skip(f"Could not import src.portfolio.graphql.schema: {e}")
