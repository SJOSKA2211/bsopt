import importlib

import pytest


def test_import_pricing_quant_utils():
    # Automatically generated import test for pricing.quant_utils
    module = importlib.import_module("src.pricing.quant_utils")
    assert module is not None

def test_initialization_pricing_quant_utils():
    # Automatically generated init test for pricing.quant_utils
    try:
        importlib.import_module("src.pricing.quant_utils")
    except Exception as e:
        pytest.skip(f"Could not import src.pricing.quant_utils: {e}")
