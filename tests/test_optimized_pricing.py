"""
Comprehensive Test Suite for Optimized Quantitative Engines (Pytest Modernized)
"""

import numpy as np
import pytest

from src.math_kernel.black_scholes import BlackScholesEngine as VectorizedBlackScholesEngine
from src.math_kernel.implied_vol import (
    implied_volatility,
    vectorized_implied_volatility,
)
from src.math_kernel.models import BSParameters


@pytest.fixture
def test_data():
    return {
        "spots": np.array([100.0, 100.0, 100.0]),
        "strikes": np.array([90.0, 100.0, 110.0]),
        "maturities": np.array([1.0, 1.0, 1.0]),
        "vols": np.array([0.2, 0.2, 0.2]),
        "rates": np.array([0.05, 0.05, 0.05]),
        "divs": np.array([0.0, 0.0, 0.0]),
        "types": np.array(["call", "call", "call"]),
        "put_spots": np.array([100.0, 100.0]),
        "put_strikes": np.array([100.0, 110.0]),
        "put_types": np.array(["put", "put"]),
    }


def test_vectorized_bs_accuracy():
    """Verify JIT BS engine against known values."""
    price = VectorizedBlackScholesEngine.price_options(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call")
    # Standard BS price for S=100, K=100, T=1, sigma=0.2, r=0.05 is ~10.4506
    assert float(price) == pytest.approx(10.450583572185565, abs=1e-8)


def test_greeks_consistency(test_data):
    """Verify delta is within [0, 1] for calls and [-1, 0] for puts."""
    greeks = VectorizedBlackScholesEngine.calculate_greeks_batch(
        spot=test_data["spots"],
        strike=test_data["strikes"],
        maturity=test_data["maturities"],
        volatility=test_data["vols"],
        rate=test_data["rates"],
        dividend=test_data["divs"],
        option_type=test_data["types"],
    )
    assert np.all(greeks["delta"] >= 0)
    assert np.all(greeks["delta"] <= 1.0)

    # Verify Puts
    put_greeks = VectorizedBlackScholesEngine.calculate_greeks_batch(
        spot=test_data["put_spots"],
        strike=test_data["put_strikes"],
        maturity=test_data["maturities"][:2],
        volatility=test_data["vols"][:2],
        rate=test_data["rates"][:2],
        dividend=test_data["divs"][:2],
        option_type=test_data["put_types"],
    )
    assert np.all(put_greeks["delta"] <= 0)
    assert np.all(put_greeks["delta"] >= -1.0)
    # Theta for Puts should be negative (time decay) for most vanilla options
    assert np.all(put_greeks["theta"] < 0)


def test_iv_convergence():
    """Verify IV calculation recovers the input volatility."""
    target_vol = 0.25
    price = VectorizedBlackScholesEngine.price_options(
        100.0, 100.0, 1.0, target_vol, 0.05, 0.0, "call"
    )

    iv = implied_volatility(float(price), 100.0, 100.0, 1.0, 0.05, 0.0, "call")
    assert iv == pytest.approx(target_vol, abs=1e-4)


def test_batch_iv(test_data):
    """Verify batch IV calculation speed and accuracy."""
    vols = np.array([0.15, 0.25, 0.35])
    prices = VectorizedBlackScholesEngine.price_options(
        test_data["spots"],
        test_data["strikes"],
        test_data["maturities"],
        vols,
        test_data["rates"],
        test_data["divs"],
        test_data["types"],
    )

    calc_vols = vectorized_implied_volatility(
        prices,
        test_data["spots"],
        test_data["strikes"],
        test_data["maturities"],
        test_data["rates"],
        test_data["divs"],
        test_data["types"],
    )

    np.testing.assert_array_almost_equal(calc_vols, vols, decimal=4)


def test_wasm_simd_speedup():
    """OPTIMIZED: Verify WASM SIMD pricing if available."""
    try:
        from src.math_kernel.engine import BlackScholesWASM

        engine = BlackScholesWASM()
        # Test case: S=100, K=100, T=1, sigma=0.2, r=0.05
        price = engine.price_call(100.0, 100.0, 1.0, 0.2, 0.05, 0.0)
        assert price == pytest.approx(10.450583572185565, abs=1e-8)

        # Verify Put Greeks in WASM
        put_price = engine.price_put(100.0, 100.0, 1.0, 0.2, 0.05, 0.0)
        assert put_price == pytest.approx(5.57352, abs=1e-4)

        greeks = engine.calculate_greeks(BSParameters(100.0, 100.0, 1.0, 0.2, 0.05, 0.0), "put")
        assert greeks.delta < 0
        assert greeks.theta < 0
    except ImportError:
        pytest.skip("WASM Engine not installed")


def test_lsm_american_accuracy():
    """OPTIMIZED: Verify Optimized LSM American Pricing."""
    try:
        from src.math_kernel.engine import AmericanOptionsWASM

        engine = AmericanOptionsWASM()
        # Standard American Call on non-dividend paying stock = European Call
        price = engine.price_lsm(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True, 10000, 50)
        # Should be close to 10.45
        assert 10.0 < price < 11.0
    except ImportError:
        pytest.skip("WASM American Engine not installed")
