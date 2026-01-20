import numpy as np

from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.pricing.quant_utils import batch_greeks_jit


def test_batch_greeks_vs_single():
    """Verify batch Greeks match single Greeks calculation."""
    spots = np.array([100.0, 110.0, 90.0])
    strikes = np.array([100.0, 100.0, 100.0])
    maturities = np.array([1.0, 0.5, 2.0])
    vols = np.array([0.2, 0.25, 0.3])
    rates = np.array([0.05, 0.04, 0.06])
    divs = np.array([0.02, 0.01, 0.03])
    is_call = np.array([True, False, True])

    delta, gamma, vega, theta, rho = batch_greeks_jit(
        spots, strikes, maturities, vols, rates, divs, is_call
    )

    for i in range(len(spots)):
        params = BSParameters(
            spot=spots[i],
            strike=strikes[i],
            maturity=maturities[i],
            volatility=vols[i],
            rate=rates[i],
            dividend=divs[i],
        )
        option_type = "call" if is_call[i] else "put"
        expected = BlackScholesEngine.calculate_greeks(params, option_type)

        # expected is an OptionGreeks object
        assert np.isclose(delta[i], expected.delta, rtol=1e-5)
        assert np.isclose(gamma[i], expected.gamma, rtol=1e-5)
        assert np.isclose(vega[i], expected.vega, rtol=1e-5)
        assert np.isclose(theta[i], expected.theta, rtol=1e-5)
        assert np.isclose(rho[i], expected.rho, rtol=1e-5)


def test_batch_greeks_edge_cases():
    """Test batch Greeks with near-zero maturity."""
    spots = np.array([100.0])
    strikes = np.array([100.0])
    maturities = np.array([1e-8])  # Near zero
    vols = np.array([0.2])
    rates = np.array([0.05])
    divs = np.array([0.02])
    is_call = np.array([True])

    delta, gamma, vega, theta, rho = batch_greeks_jit(
        spots, strikes, maturities, vols, rates, divs, is_call
    )

    assert not np.isnan(delta).any()
    assert not np.isinf(delta).any()
    assert delta[0] >= 0
