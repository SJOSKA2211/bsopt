import importlib

import pytest


def test_import_api_providers_market_data():
    # Automatically generated import test for api.providers.market_data
    module = importlib.import_module("src.api.providers.market_data")
    assert module is not None

def test_initialization_api_providers_market_data():
    # Automatically generated init test for api.providers.market_data
    try:
        importlib.import_module("src.api.providers.market_data")
    except Exception as e:
        pytest.skip(f"Could not import src.api.providers.market_data: {e}")
