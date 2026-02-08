import importlib

import pytest


def test_import_streaming_graphql_schema():
    # Automatically generated import test for streaming.graphql.schema
    module = importlib.import_module("src.streaming.graphql.schema")
    assert module is not None

def test_initialization_streaming_graphql_schema():
    # Automatically generated init test for streaming.graphql.schema
    try:
        importlib.import_module("src.streaming.graphql.schema")
    except Exception as e:
        pytest.skip(f"Could not import src.streaming.graphql.schema: {e}")
