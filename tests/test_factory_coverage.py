import pytest
from src.pricing.factory import PricingEngineFactory
from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.monte_carlo import MonteCarloEngine
from src.pricing.lattice import BinomialTreePricer
from src.pricing.finite_difference import CrankNicolsonSolver

def test_factory_get_strategy():
    assert isinstance(PricingEngineFactory.get_strategy("black_scholes"), BlackScholesEngine)
    assert isinstance(PricingEngineFactory.get_strategy("monte_carlo"), MonteCarloEngine)
    assert isinstance(PricingEngineFactory.get_strategy("binomial"), BinomialTreePricer)
    assert isinstance(PricingEngineFactory.get_strategy("fdm"), CrankNicolsonSolver)
    
    # Case insensitivity
    assert isinstance(PricingEngineFactory.get_strategy("BLACK_SCHOLES"), BlackScholesEngine)

def test_factory_invalid_model():
    with pytest.raises(ValueError, match="Unknown pricing model"):
        PricingEngineFactory.get_strategy("invalid_model")
