from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.ml.reinforcement_learning.gnn_policy import GNNFeatureExtractor, SACGNNPolicy
from src.ml.reinforcement_learning.online_agent import OnlineRLAgent


@pytest.fixture
def rl_agent():
    with patch("src.ml.reinforcement_learning.online_agent.TD3") as mock_td3:
        mock_td3.load.return_value = MagicMock()
        agent = OnlineRLAgent(model_path="mock_model", initial_balance=100000)
        return agent


def test_agent_init(rl_agent):
    assert rl_agent.balance == 100000
    assert len(rl_agent.positions) == 10
    assert np.all(rl_agent.positions == 0)


def test_get_state_vector(rl_agent):
    market_data = {
        "prices": np.random.uniform(150, 160, 10),
        "strikes": np.random.uniform(145, 155, 10),
        "greeks": np.random.normal(0, 1, (10, 5)),
        "indicators": np.random.normal(0, 1, 20),
    }
    state = rl_agent._get_state_vector(market_data)
    assert len(state) == 100
    assert state[0] == 1.0  # balance / initial_balance


def test_process_market_data(rl_agent):
    market_data = {
        "prices": np.random.uniform(150, 160, 10),
        "strikes": np.random.uniform(145, 155, 10),
        "greeks": np.random.normal(0, 1, (10, 5)),
        "indicators": np.random.normal(0, 1, 20),
    }
    rl_agent.model.predict.return_value = (np.random.uniform(-1, 1, 10), None)
    action = rl_agent.process_market_data(market_data)
    assert len(action) == 10
    assert rl_agent.last_state is not None
    assert rl_agent.last_action is not None


def test_calculate_reward(rl_agent):
    market_data = {"prices": np.random.uniform(150, 160, 10)}
    # First call initializes prev_value
    reward1 = rl_agent._calculate_reward(market_data)
    assert reward1 == 0.0

    # Second call calculates return
    rl_agent.balance = 101000  # 1% gain
    reward2 = rl_agent._calculate_reward(market_data)
    assert reward2 > 0


def test_store_transition(rl_agent):
    state = np.zeros(100)
    action = np.zeros(10)
    reward = 1.0
    next_state = np.zeros(100)

    rl_agent._store_transition(state, action, reward, next_state)
    assert rl_agent._buffer_idx == 1
    assert np.all(rl_agent._obs_buffer[0] == state)


# GNN Policy Tests
def test_gnn_feature_extractor():
    input_dim = 10
    hidden_dim = 16
    output_dim = 8
    model = GNNFeatureExtractor(input_dim, hidden_dim, output_dim)

    x = torch.randn(5, input_dim)  # 5 nodes
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    output = model(x, edge_index)
    assert output.shape == (5, output_dim)


def test_sac_gnn_policy():
    state_dim = 10
    action_dim = 4
    model = SACGNNPolicy(state_dim, action_dim)

    x = torch.randn(5, state_dim)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    output = model(x, edge_index)
    assert output.shape == (5, action_dim * 2)