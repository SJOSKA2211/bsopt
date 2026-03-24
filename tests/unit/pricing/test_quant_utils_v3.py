import numpy as np

from src.math_kernel import quant_utils

# Set a consistent seed for reproducibility
np.random.seed(42)

def test_corrado_miller_initial_guess():
    """
    Tests the Corrado-Miller initial IV guess with a single option.
    This function is JIT-compiled, so we test its output on known inputs.
    """
    # Arrange: Single option data as arrays
    market_price = np.array([10.0])
    spot = np.array([100.0])
    strike = np.array([105.0])
    maturity = np.array([0.25])  # 3 months
    rate = np.array([0.05])
    dividend = np.array([0.0])
    option_type = np.array([0])  # 0 for call

    # Act
    iv_guess = quant_utils.corrado_miller_initial_guess(
        market_price, spot, strike, maturity, rate, dividend, option_type
    )

    # Assert: Check shape and plausibility
    assert iv_guess.shape == (1,)
    # For these params, a volatility around 30-60% is reasonable.
    assert 0.1 < iv_guess[0] < 1.0

def test_batch_bs_price_jit_call():
    """
    Tests the JIT-compiled Black-Scholes pricing kernel for a batch of call options.
    """
    # Arrange: Batch of 3 identical options for simplicity
    n = 3
    S = np.full(n, 100.0)
    K = np.full(n, 105.0)
    T = np.full(n, 0.25)
    sigma = np.full(n, 0.4)  # 40% volatility
    r = np.full(n, 0.05)
    q = np.full(n, 0.0)
    is_call = np.ones(n)  # 1 for call

    # Act
    prices = quant_utils.batch_bs_price_jit(S, K, T, sigma, r, q, is_call)

    # Assert: Known result for these parameters is ~7.11
    assert prices.shape == (n,)
    assert np.allclose(prices, 7.11, atol=0.02)

def test_batch_bs_price_jit_put():
    """
    Tests the JIT-compiled Black-Scholes pricing kernel for a batch of put options.
    """
    # Arrange
    n = 3
    S = np.full(n, 100.0)
    K = np.full(n, 105.0)
    T = np.full(n, 0.25)
    sigma = np.full(n, 0.4)
    r = np.full(n, 0.05)
    q = np.full(n, 0.0)
    is_call = np.zeros(n)  # 0 for put

    # Act
    prices = quant_utils.batch_bs_price_jit(S, K, T, sigma, r, q, is_call)

    # Assert: Known result for these parameters is ~10.80
    assert prices.shape == (n,)
    assert np.allclose(prices, 10.80, atol=0.02)
