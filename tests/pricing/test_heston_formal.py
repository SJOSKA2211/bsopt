import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.pricing.models.heston_fft import batch_heston_price_jit

# 🚀 SINGULARITY: Formal Verification of Pricing Kernels
# Verifies that our JIT kernels are numerically stable across the universe of inputs.


@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
@given(
    spot=st.floats(min_value=10.0, max_value=1000.0),
    strike=st.floats(min_value=10.0, max_value=1000.0),
    maturity=st.floats(min_value=0.01, max_value=10.0),
    rate=st.floats(min_value=0.0, max_value=0.2),
    v0=st.floats(min_value=0.01, max_value=1.0),
    kappa=st.floats(min_value=0.1, max_value=5.0),
    theta=st.floats(min_value=0.01, max_value=1.0),
    sigma=st.floats(min_value=0.1, max_value=1.0),
    rho=st.floats(min_value=-0.99, max_value=0.99),
    is_call=st.booleans(),
)
def test_heston_fft_stability(
    spot, strike, maturity, rate, v0, kappa, theta, sigma, rho, is_call
):
    """🚀 SOTA: Property-based testing for Heston stability."""

    # 1. Setup inputs for vectorized kernel
    spots = np.array([spot], dtype=np.float64)
    strikes = np.array([strike], dtype=np.float64)
    maturities = np.array([maturity], dtype=np.float64)
    rates = np.array([rate], dtype=np.float64)
    v0s = np.array([v0], dtype=np.float64)
    kappas = np.array([kappa], dtype=np.float64)
    thetas = np.array([theta], dtype=np.float64)
    sigmas = np.array([sigma], dtype=np.float64)
    rhos = np.array([rho], dtype=np.float64)
    is_calls = np.array([is_call], dtype=bool)
    out = np.empty(1, dtype=np.float64)

    # 2. Execute JIT kernel
    batch_heston_price_jit(
        spots,
        strikes,
        maturities,
        rates,
        v0s,
        kappas,
        thetas,
        sigmas,
        rhos,
        is_calls,
        out,
    )

    price = out[0]

    # 3. Assertions (The Laws of Physics)
    assert not np.isnan(price), f"Price is NaN for {spot}, {strike}"
    assert not np.isinf(price), f"Price is Inf for {spot}, {strike}"
    assert price >= 0, f"Negative price {price} for {spot}, {strike}"

    # Arbitrage bound: Call price cannot exceed spot
    if is_call:
        assert price <= spot + 1e-5, f"Call price {price} exceeds spot {spot}"

    # Boundary: Intrinsic value bound
    intrinsic = max(spot - strike, 0) if is_call else max(strike - spot, 0)
    assert price >= intrinsic - 1e-5, f"Price {price} below intrinsic {intrinsic}"


if __name__ == "__main__":
    test_heston_fft_stability()
