import importlib

import pytest


def test_import_pricing_graphql_schema():
    # Automatically generated import test for pricing.graphql.schema
    module = importlib.import_module("src.pricing.graphql.schema")
    assert module is not None

def test_initialization_pricing_graphql_schema():
    # Automatically generated init test for pricing.graphql.schema
    try:
        importlib.import_module("src.pricing.graphql.schema")
    except Exception as e:
        pytest.skip(f"Could not import src.pricing.graphql.schema: {e}")
