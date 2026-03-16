import os
import sys
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# Note: We will implement TradingEnvironment in src/ml/reinforcement_learning/trading_env.py
# For TDD, we write the tests first and expect them to fail until implementation


def test_trading_env_initialization():
    """Verify that the environment initializes with correct spaces."""
    from services.ml.reinforcement_learning.trading_env import TradingEnvironment

    mock_data_provider = MagicMock()
    window_size = 5
    env = TradingEnvironment(data_provider=mock_data_provider, window_size=window_size)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (window_size, 100)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (10,)
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0


def test_trading_env_reset():
    """Verify that reset returns a valid initial observation."""
    from services.ml.reinforcement_learning.trading_env import TradingEnvironment

    mock_data_provider = MagicMock()
    window_size = 5
    # Mock initial data
    mock_data = {
        "prices": np.random.rand(10),
        "greeks": np.random.rand(10, 5),
        "indicators": np.random.rand(20),
    }
    mock_data_provider.get_latest_data.return_value = mock_data

    env = TradingEnvironment(data_provider=mock_data_provider, window_size=window_size)
    obs, info = env.reset()

    assert obs.shape == (window_size, 100)
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert env.current_step == 0


def test_trading_env_step():
    """Verify that step function processes actions and returns expected outputs."""
    from services.ml.reinforcement_learning.trading_env import TradingEnvironment

    mock_data_provider = MagicMock()
    window_size = 5
    # Mock data for steps
    mock_data = {
        "prices": np.ones(10) * 100.0,
        "greeks": np.zeros((10, 5)),
        "indicators": np.zeros(20),
    }
    mock_data_provider.get_latest_data.return_value = mock_data
    mock_data_provider.get_data_at_step.return_value = mock_data
    mock_data_provider.__len__.return_value = 100

    env = TradingEnvironment(data_provider=mock_data_provider, window_size=window_size)
    env.reset()

    action = np.ones(10) * 0.01  # Target 1% weight for all (total 10%)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (window_size, 100)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.current_step == 1
    # portfolio_value ~ 100000, price = 100, weight = 0.01 -> units = 10
    assert np.allclose(env.positions, np.ones(10) * 10.0)


def test_trading_env_drawdown_truncation():
    """Verify that the environment truncates on excessive drawdown."""
    from services.ml.reinforcement_learning.trading_env import TradingEnvironment

    mock_data_provider = MagicMock()
    env = TradingEnvironment(data_provider=mock_data_provider, initial_balance=1000)
    env.reset()

    # Simulate a massive loss
    env.balance = 400  # 60% loss

    action = np.zeros(10)
    _, _, _, truncated, _ = env.step(action)

    assert truncated is True
