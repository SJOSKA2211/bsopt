import pytest
from src.pricing.factory import PricingEngineFactory
from src.pricing.black_scholes import BlackScholesEngine
from src.pricing.monte_carlo import MonteCarloEngine
from src.pricing.lattice import BinomialTreePricer
from src.pricing.finite_difference import CrankNicolsonSolver

def test_factory_get_black_scholes():
    strategy = PricingEngineFactory.get_strategy("black_scholes")
    assert isinstance(strategy, BlackScholesEngine)

def test_factory_get_monte_carlo():
    strategy = PricingEngineFactory.get_strategy("monte_carlo")
    assert isinstance(strategy, MonteCarloEngine)

def test_factory_get_binomial():
    strategy = PricingEngineFactory.get_strategy("binomial")
    assert isinstance(strategy, BinomialTreePricer)

def test_factory_get_fdm():
    strategy = PricingEngineFactory.get_strategy("fdm")
    assert isinstance(strategy, CrankNicolsonSolver)

def test_factory_get_unknown():
    with pytest.raises(ValueError, match="Unknown pricing model: unknown"):
        PricingEngineFactory.get_strategy("unknown")

def test_factory_case_insensitive():
    strategy = PricingEngineFactory.get_strategy("BLACK_SCHOLES")
    assert isinstance(strategy, BlackScholesEngine)
