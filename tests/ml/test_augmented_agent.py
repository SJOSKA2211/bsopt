from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ml.reinforcement_learning.augmented_agent import AugmentedRLAgent


@pytest.fixture
def agent_config():
    return {
        "price_state_dim": 10,
        "sentiment_state_dim": 1,
        "action_dim": 3
    }

def test_augmented_agent_initialization(agent_config):
    """Test that AugmentedRLAgent initializes with correct observation space."""
    agent = AugmentedRLAgent(config=agent_config)
    # Total observation space should be price_state + sentiment_state
    assert agent.observation_dim == 11
    assert agent.action_dim == 3

def test_observation_concatenation(agent_config):
    """Test that Price_State and Sentiment_State are correctly concatenated."""
    agent = AugmentedRLAgent(config=agent_config)
    
    price_state = np.random.randn(10)
    sentiment_score = 0.75
    
    observation = agent.get_augmented_observation(price_state, sentiment_score)
    
    assert len(observation) == 11
    assert observation[-1] == 0.75 # Sentiment should be the last element
    assert np.array_equal(observation[:10], price_state)

def test_agent_act_no_model(agent_config):
    """Test that agent produces random actions when no model is loaded."""
    agent = AugmentedRLAgent(config=agent_config)
    assert agent.model is None
    
    observation = np.random.randn(11)
    action = agent.act(observation)
    
    assert len(action) == 3
    assert np.all(action >= -1) and np.all(action <= 1)

def test_agent_act_with_augmented_state(agent_config):
    """Test that agent can produce an action from the augmented state."""
    agent = AugmentedRLAgent(config=agent_config)
    agent.model = MagicMock()
    
    # Mock model prediction
    mock_action = np.array([0.1, 0.2, 0.7])
    agent.model.predict.return_value = (mock_action, None)
    
    observation = np.random.randn(11)
    action = agent.act(observation)
    
    assert len(action) == 3
    agent.model.predict.assert_called_once()
