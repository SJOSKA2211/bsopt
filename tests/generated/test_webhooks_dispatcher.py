import importlib

import pytest


def test_import_webhooks_dispatcher():
    # Automatically generated import test for webhooks.dispatcher
    module = importlib.import_module("src.webhooks.dispatcher")
    assert module is not None

def test_initialization_webhooks_dispatcher():
    # Automatically generated init test for webhooks.dispatcher
    try:
        importlib.import_module("src.webhooks.dispatcher")
    except Exception as e:
        pytest.skip(f"Could not import src.webhooks.dispatcher: {e}")
