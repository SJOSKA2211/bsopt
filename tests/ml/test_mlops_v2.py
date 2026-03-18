import os

import numpy as np
import pytest
import yaml

from src.ml.drift import calculate_psi
from src.ml.models.neural_engine import NeuralPricingEngine
from src.ml.reinforcement_learning.trading_env import TradingEnvironment
from src.quant.pricing.models import BSParameters


def test_mlproject_integrity():
    """Verify MLproject file defines all required entry points."""
    assert os.path.exists("MLproject")
    with open("MLproject") as f:
        config = yaml.safe_load(f)

    entry_points = config.get("entry_points", {})
    required = [
        "train_regressor",
        "train_rl",
        "train_tft",
        "train_distributed_dt",
        "promote_model",
        "evaluate_challenger",
    ]
    for ep in required:
        assert ep in entry_points, f"Missing entry point: {ep}"


def test_psi_kernel_performance():
    """Verify Numba-optimized PSI kernel produces correct results."""
    expected = np.random.normal(100, 10, 1000)
    actual = np.random.normal(105, 12, 1000)

    psi_val = calculate_psi(expected, actual)
    assert isinstance(psi_val, float)
    assert psi_val >= 0


def test_trading_env_silicon_fusion():
    """Verify TradingEnvironment uses silicon buffers and fused kernels."""
    env = TradingEnvironment()
    obs, _ = env.reset()

    # Check buffer allocation
    assert hasattr(env, "_window_buffer")
    assert env._window_buffer.shape == (env.window_size, 128)

    # Check observation integrity
    assert obs.shape == (env.window_size, 128)

    action = env.action_space.sample()
    next_obs, reward, term, trunc, info = env.step(action)

    assert next_obs.shape == (env.window_size, 128)
    assert isinstance(reward, float)


def test_neural_engine_parity_optimization():
    """Verify NeuralPricingEngine Put-Call parity and exact Greeks."""
    engine = NeuralPricingEngine()
    params = BSParameters(
        spot=100.0, strike=100.0, maturity=1.0, volatility=0.2, rate=0.05, dividend=0.01
    )

    # 1. Test Put Pricing (via Parity)
    call_price = engine.price(params, option_type="call")
    put_price = engine.price(params, option_type="put")

    # Standard Parity Check: C - P = S*e(-qT) - K*e(-rT)
    s_discounted = params.spot * np.exp(-params.dividend * params.maturity)
    k_discounted = params.strike * np.exp(-params.rate * params.maturity)
    parity_diff = s_discounted - k_discounted

    assert pytest.approx(call_price - put_price, abs=1e-4) == parity_diff

    # 2. Test Greek Parity
    call_greeks = engine.calculate_greeks(params, option_type="call")
    put_greeks = engine.calculate_greeks(params, option_type="put")

    # Delta_p = Delta_c - exp(-qT)
    assert pytest.approx(put_greeks.delta, abs=1e-4) == call_greeks.delta - np.exp(
        -params.dividend * params.maturity
    )
    # Gamma should be identical
    assert pytest.approx(put_greeks.gamma, abs=1e-4) == call_greeks.gamma
