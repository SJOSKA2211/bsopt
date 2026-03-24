import pytest

import numpy as np

from src.quant.pricing.quant_utils import (
    batch_bs_price_jit,
    batch_greeks_jit,
    corrado_miller_initial_guess,
    heston_char_func_jit,
    jit_cn_solver,
    jit_generate_log_paths,
    jit_generate_paths,
    jit_lsm_american,
    jit_mc_european_price,
    jit_mc_european_price_and_greeks,
    jit_mc_european_with_control_variate,
    scalar_bs_price_jit,
    scalar_greeks_jit,
    thomas_algorithm,
    vectorized_newton_raphson_iv_jit,
)


class TestQuantUtils:
    def test_jit_generate_paths(self):
        S0, T, r, sigma, q = 100.0, 1.0, 0.05, 0.2, 0.0
        paths, steps = 100, 50
        result = jit_generate_paths(S0, T, r, sigma, q, paths, steps)
        assert result.shape == (paths, steps + 1)

    def test_jit_generate_log_paths(self):
        S0, T, r, sigma, q = 100.0, 1.0, 0.05, 0.2, 0.0
        paths, steps = 100, 50
        result = jit_generate_log_paths(S0, T, r, sigma, q, paths, steps)
        assert result.shape == (steps + 1, paths)

    def test_batch_bs_price_jit(self):
        S, K, T, sigma, r, q = (
            np.array([100.0]),
            np.array([100.0]),
            np.array([1.0]),
            np.array([0.2]),
            np.array([0.05]),
            np.array([0.0]),
        )
        is_call = np.array([True])
        price = batch_bs_price_jit(S, K, T, sigma, r, q, is_call)
        self.assertGreater(price[0], 0)

        # Test T < 1e-7
        T_zero = np.array([0.0])
        price_zero = batch_bs_price_jit(S, K, T_zero, sigma, r, q, is_call)
        assert price_zero[0] == 0.0

    def test_batch_greeks_jit(self):
        S, K, T, sigma, r, q = (
            np.array([100.0]),
            np.array([100.0]),
            np.array([1.0]),
            np.array([0.2]),
            np.array([0.05]),
            np.array([0.0]),
        )
        is_call = np.array([True])
        delta, gamma, vega, theta, rho = batch_greeks_jit(S, K, T, sigma, r, q, is_call)
        self.assertGreater(delta[0], 0)

    def test_thomas_algorithm(self):
        lower = np.array([1.0, 1.0])
        diag = np.array([4.0, 4.0, 4.0])
        upper = np.array([1.0, 1.0])
        rhs = np.array([1.0, 2.0, 3.0])
        x = thomas_algorithm(lower, diag, upper, rhs)
        assert len(x) == 3

    def test_jit_cn_solver(self):
        s_grid = np.linspace(0, 200, 50)
        V = jit_cn_solver(s_grid, 100.0, 1.0, 0.05, 0.2, 0.0, True, 10)
        assert len(V) == 50

    def test_vectorized_newton_raphson_iv_jit(self):
        market_prices = np.array([10.45])
        S, K, T, r, q = (
            np.array([100.0]),
            np.array([100.0]),
            np.array([1.0]),
            np.array([0.05]),
            np.array([0.0]),
        )
        is_call = np.array([True])
        sigma_init = np.array([0.2])
        iv = vectorized_newton_raphson_iv_jit(market_prices, S, K, T, r, q, is_call, sigma_init)
        assert iv[0] == pytest.approx(0.2, delta=0.1)

    def test_heston_char_func_jit(self):
        res = heston_char_func_jit(1.0, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
        assert isinstance(res, complex)

    def test_jit_mc_european_price(self):
        p, s = jit_mc_european_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 100, True, True)
        self.assertGreater(p, 0)

    def test_jit_mc_european_price_and_greeks(self):
        res = jit_mc_european_price_and_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 100, True, True)
        assert len(res) == 5

    def test_jit_mc_european_with_control_variate(self):
        p, s = jit_mc_european_with_control_variate(
            100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 100, True, True
        )
        self.assertGreater(p, 0)

    def test_jit_lsm_american(self):
        p = jit_lsm_american(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 100, 10, True)
        self.assertGreater(p, 0)

    def test_scalar_bs_price_jit(self):
        p = scalar_bs_price_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True)
        self.assertGreater(p, 0)
        # Test T < 1e-7
        p_zero = scalar_bs_price_jit(100.0, 100.0, 0.0, 0.2, 0.05, 0.0, True)
        assert p_zero == 0.0

    def test_scalar_greeks_jit(self):
        res = scalar_greeks_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, True)
        assert len(res) == 5
        res_put = scalar_greeks_jit(100.0, 100.0, 1.0, 0.2, 0.05, 0.0, False)
        assert len(res_put) == 5

    def test_corrado_miller_initial_guess(self):
        S, K, T, r, q = (
            np.array([100.0]),
            np.array([100.0]),
            np.array([1.0]),
            np.array([0.05]),
            np.array([0.0]),
        )
        market_price = np.array([10.45])
        option_type = np.array([0])  # Call
        iv = corrado_miller_initial_guess(market_price, S, K, T, r, q, option_type)
        self.assertGreater(iv[0], 0)



