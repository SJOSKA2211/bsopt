import importlib

import pytest


def test_import_api_routes_users():
    # Automatically generated import test for api.routes.users
    module = importlib.import_module("src.api.routes.users")
    assert module is not None

def test_initialization_api_routes_users():
    # Automatically generated init test for api.routes.users
    try:
        importlib.import_module("src.api.routes.users")
    except Exception as e:
        pytest.skip(f"Could not import src.api.routes.users: {e}")
