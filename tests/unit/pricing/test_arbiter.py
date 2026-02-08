import numpy as np
import pytest

from src.pricing.arbiter import EngineArbiter, PricingModel, PricingRequest
from src.pricing.models import BSParameters


@pytest.fixture
def arbiter():
    return EngineArbiter()

@pytest.fixture
def sample_params():
    return BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, 
        volatility=0.2, rate=0.05, dividend=0.0
    )

def test_route_explicit_bs(arbiter, sample_params):
    req = PricingRequest(params=sample_params, model=PricingModel.BLACK_SCHOLES)
    price = arbiter.route_request(req)
    assert price > 0

def test_route_explicit_mc(arbiter, sample_params):
    req = PricingRequest(params=sample_params, model=PricingModel.MONTE_CARLO, engine_config={"n_paths": 1000})
    price = arbiter.route_request(req)
    assert price > 0

def test_route_smart_american(arbiter, sample_params):
    # Should fallback to MC if WASM is unavailable
    req = PricingRequest(params=sample_params, style="american")
    price = arbiter.route_request(req)
    assert price > 0

def test_route_batch(arbiter):
    S = np.array([100.0, 110.0])
    K = np.array([100.0, 100.0])
    T = np.array([1.0, 1.0])
    sigma = np.array([0.2, 0.2])
    r = np.array([0.05, 0.05])
    is_call = np.array([True, True])
    
    prices = arbiter.route_batch(S, K, T, sigma, r, is_call)
    assert len(prices) == 2
    assert np.all(prices > 0)

def test_route_explicit_wasm_fallback(arbiter, sample_params):
    # If WASM instance is None, should fallback to BS
    arbiter.wasm_engine.instance = None
    req = PricingRequest(params=sample_params, model=PricingModel.WASM)
    price = arbiter.route_request(req)
    assert price > 0
