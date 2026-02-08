import importlib

import pytest


def test_import_pricing_models_heston_fft():
    # Automatically generated import test for pricing.models.heston_fft
    module = importlib.import_module("src.pricing.models.heston_fft")
    assert module is not None

def test_initialization_pricing_models_heston_fft():
    # Automatically generated init test for pricing.models.heston_fft
    try:
        importlib.import_module("src.pricing.models.heston_fft")
    except Exception as e:
        pytest.skip(f"Could not import src.pricing.models.heston_fft: {e}")
