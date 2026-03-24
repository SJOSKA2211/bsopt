import pytest

from src.math_kernel.models import (
    BSParameters,
    HestonParams,
    OptionGreeks,
    global_model_pool,
)


def test_bs_parameters_validation():
    # Valid
    p = BSParameters(spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05)
    assert p.spot == 100

    # Invalid spot
    with pytest.raises(ValueError):
        BSParameters(spot=-1, strike=100, maturity=1, volatility=0.2, rate=0.05)


def test_option_greeks():
    g = OptionGreeks(delta=0.5, gamma=0.02, theta=-0.01, vega=0.1, rho=0.05)
    assert g["delta"] == 0.5
    assert "gamma" in g


def test_heston_params_validation():
    # Valid (Feller condition: 2 * 2 * 0.04 > 0.1^2 => 0.16 > 0.01)
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=-0.7)
    assert p.v0 == 0.04

    # Feller violation (2 * 1 * 0.04 < 0.5^2 => 0.08 < 0.25)
    with pytest.raises(ValueError):
        HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7)


def test_model_pool():
    p1 = global_model_pool.get_bs_params(
        spot=100, strike=100, maturity=1, volatility=0.2, rate=0.05
    )
    global_model_pool.release_bs_params(p1)
    p2 = global_model_pool.get_bs_params(
        spot=110, strike=100, maturity=1, volatility=0.2, rate=0.05
    )
    assert p1 is p2  # Should be same object from pool
    assert p2.spot == 110
