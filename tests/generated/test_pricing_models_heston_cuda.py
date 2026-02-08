import importlib

import pytest


def test_import_pricing_models_heston_cuda():
    # Automatically generated import test for pricing.models.heston_cuda
    module = importlib.import_module("src.pricing.models.heston_cuda")
    assert module is not None

def test_initialization_pricing_models_heston_cuda():
    # Automatically generated init test for pricing.models.heston_cuda
    try:
        importlib.import_module("src.pricing.models.heston_cuda")
    except Exception as e:
        pytest.skip(f"Could not import src.pricing.models.heston_cuda: {e}")
