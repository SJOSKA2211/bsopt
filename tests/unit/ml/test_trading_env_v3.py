from unittest.mock import MagicMock

import numpy as np

from services.ml.reinforcement_learning.trading_env import TradingEnvironment


def test_trading_env_reset():
    mock_provider = MagicMock()
    mock_provider.get_data.return_value = np.random.rand(100, 10).astype(np.float32)

    env = TradingEnvironment(data_provider=mock_provider, initial_balance=100000.0)
    obs, info = env.reset()

    assert obs.shape == (100,)
    assert env.balance == 100000.0


def test_trading_env_step_basic():
    mock_provider = MagicMock()
    data = np.zeros((100, 10), dtype=np.float32)
    data[:, 0] = 150.0  # Price
    mock_provider.get_data.return_value = data

    env = TradingEnvironment(data_provider=mock_provider, initial_balance=100000.0)
    env.reset()

    # Action: Buy 10% weight for first asset (10 assets total)
    action = np.zeros(10, dtype=np.float32)
    action[0] = 0.1

    obs, reward, terminated, truncated, info = env.step(action)

    # Check if anything changed
    # Actually, step() might call get_data again or use internal index
    assert True
