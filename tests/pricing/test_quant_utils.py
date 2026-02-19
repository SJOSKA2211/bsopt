import numpy as np

from src.pricing import quant_utils


def test_corrado_miller():
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    r = np.array([0.05])
    q = np.array([0.0])
    price = np.array([10.45])  # Approx BS price for 20% vol
    option_type = np.array([0])  # Call

    iv = quant_utils.corrado_miller_initial_guess(price, S, K, T, r, q, option_type)
    assert 0.15 < iv[0] < 0.25


def test_thomas_algorithm():
    # Solve simple system
    # 2x - y = 1
    # -x + 2y - z = 0
    # -y + 2z = 1
    # Sol: x=1, y=1, z=1
    lower = np.array([-1.0, -1.0])
    diag = np.array([2.0, 2.0, 2.0])
    upper = np.array([-1.0, -1.0])
    rhs = np.array([1.0, 0.0, 1.0])

    x = quant_utils.thomas_algorithm(lower, diag, upper, rhs)
    assert np.allclose(x, [1.0, 1.0, 1.0])


def test_newton_raphson_iv():
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    r = np.array([0.05])
    q = np.array([0.0])
    market_price = np.array([10.4506])
    is_call = np.array([True])
    sigma_init = np.array([0.1])

    iv = quant_utils.vectorized_newton_raphson_iv_jit(
        market_price, S, K, T, r, q, is_call, sigma_init
    )
    assert np.isclose(iv[0], 0.2, atol=1e-4)


def test_heston_char_func():
    # Just verify it doesn't crash and returns complex
    res = quant_utils.heston_char_func_jit(
        1.0 + 0.5j, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7
    )
    assert isinstance(res, (complex, np.complex128))


def test_cn_solver():
    s_grid = np.linspace(0, 200, 21)
    res = quant_utils.jit_cn_solver(
        s_grid,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        dividend=0.0,
        is_call=True,
        n_time=10,
    )
    assert len(res) == 21
    assert res[-1] > 0  # ITM path has value


def test_warmup_jit():
    # Ensure warmup doesn't crash
    quant_utils.warmup_jit()
    assert True


def test_batch_bs_price_edge_case():
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([0.0])  # Zero maturity
    sigma = np.array([0.2])
    r = np.array([0.05])
    q = np.array([0.0])
    is_call = np.array([True])

    prices = quant_utils.batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
    assert prices[0] == 0.0  # At the money payoff is 0

    S_itm = np.array([110.0])
    prices_itm = quant_utils.batch_bs_price_jit(S_itm, K, T, sigma, r, q, is_call)
    assert prices_itm[0] == 10.0
