import pytest

from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.factory import PricingEngineFactory
from src.pricing.finite_difference import CrankNicolsonSolver
from src.pricing.lattice import BinomialTreePricer
from src.pricing.monte_carlo import MonteCarloEngine
from src.pricing.wasm_engine import WASM_AVAILABLE, WASMPricingEngine


def test_factory_get_strategy():
    assert isinstance(
        PricingEngineFactory.get_strategy("black_scholes"), BlackScholesEngine
    )

    mc_strategy = PricingEngineFactory.get_strategy("monte_carlo")
    if WASM_AVAILABLE:
        assert isinstance(mc_strategy, WASMPricingEngine)
        assert mc_strategy.model == "monte_carlo"
    else:
        assert isinstance(mc_strategy, MonteCarloEngine)

    assert isinstance(PricingEngineFactory.get_strategy("binomial"), BinomialTreePricer)

    fdm_strategy = PricingEngineFactory.get_strategy("fdm")
    if WASM_AVAILABLE:
        assert isinstance(fdm_strategy, WASMPricingEngine)
        assert fdm_strategy.model == "fdm"
    else:
        assert isinstance(fdm_strategy, CrankNicolsonSolver)

    # Case insensitivity
    assert isinstance(
        PricingEngineFactory.get_strategy("BLACK_SCHOLES"), BlackScholesEngine
    )


def test_factory_invalid_model():
    with pytest.raises(ValueError, match="Unknown pricing model"):
        PricingEngineFactory.get_strategy("invalid_model")
