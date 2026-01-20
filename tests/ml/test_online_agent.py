import pytest
import numpy as np
import json
from unittest.mock import MagicMock, patch
from src.ml.reinforcement_learning.online_agent import OnlineRLAgent

@pytest.fixture
def mock_td3():
    with patch("src.ml.reinforcement_learning.online_agent.TD3") as MockTD3:
        mock_model = MagicMock()
        MockTD3.load.return_value = mock_model
        # predict returns (action, states)
        mock_model.predict.return_value = (np.zeros(10), None)
        yield MockTD3

@pytest.fixture
def mock_kafka():
    with patch("src.ml.reinforcement_learning.online_agent.Consumer") as MockConsumer, \
         patch("src.ml.reinforcement_learning.online_agent.Producer") as MockProducer:
        
        mock_consumer = MagicMock()
        MockConsumer.return_value = mock_consumer
        
        mock_producer = MagicMock()
        MockProducer.return_value = mock_producer
        
        yield MockConsumer, MockProducer

@pytest.fixture
def agent(mock_td3, mock_kafka):
    # Ensure TD3 is not None in the module scope by patching the import check if necessary
    # But since we patched the class, it should be truthy.
    return OnlineRLAgent(
        model_path="dummy_path",
        kafka_config={'bootstrap.servers': 'localhost:9092'}
    )

def test_init(agent):
    assert agent.model is not None
    assert agent.consumer is not None
    assert agent.producer is not None
    assert agent.initial_balance == 100000

def test_get_state_vector(agent):
    market_data = {
        'prices': [100.0] * 5, # Less than 10
        'greeks': [[0.5]*5] * 5, # Less than 10 options
        'indicators': [0.1] * 5 # Less than 20
    }
    
    state = agent._get_state_vector(market_data)
    assert state.shape == (100,)
    # Check padding works (should not crash)
    
    # Check values
    # Portfolio state (11) + Prices (10) + Greeks (50) + Indicators (20) = 91 + 9 padding = 100
    assert state[0] == 1.0 # Normalized balance

def test_process_market_data(agent):
    market_data = {'prices': [100.0] * 10}
    action = agent.process_market_data(market_data)
    
    assert len(action) == 10
    agent.model.predict.assert_called_once()
    
    # Test continuous learning triggering
    # First call sets last_state/action. Second call stores transition.
    agent.process_market_data(market_data)
    assert len(agent.experience_buffer) == 1

def test_process_market_data_no_model():
    agent = OnlineRLAgent("path")
    agent.model = None # Simulate load failure
    action = agent.process_market_data({})
    assert np.all(action == 0)

def test_run_loop(agent):
    # Mock consumer polling
    mock_msg = MagicMock()
    mock_msg.error.return_value = None
    mock_msg.value.return_value = json.dumps({'prices': [100]*10}).encode('utf-8')
    
    # To break the infinite loop, we can use side_effect on poll to return msg once then raise StopIteration or Exception
    # Or just mock poll to return msg once, then some error to break loop?
    # The loop breaks on kafka_error (non-EOF).
    
    mock_error_msg = MagicMock()
    mock_error = MagicMock()
    mock_error.code.return_value = -1 # Generic error
    mock_error_msg.error.return_value = mock_error
    
    agent.consumer.poll.side_effect = [mock_msg, mock_error_msg]
    
    agent.run()
    
    agent.producer.produce.assert_called()

def test_produce_signal_failed(agent):
    agent.producer.produce.side_effect = Exception("Kafka down")
    # Should log error but not crash
    agent._produce_signal(np.zeros(10))

def test_flush_experience(agent):
    agent.experience_buffer = [{"state": []}] * 1001
    agent._flush_experience()
    assert len(agent.experience_buffer) == 0

def test_init_without_kafka_libs():
    # Simulate missing libs
    with patch("src.ml.reinforcement_learning.online_agent.Consumer", None), \
         patch("src.ml.reinforcement_learning.online_agent.Producer", None):
        agent = OnlineRLAgent("model", kafka_config={'a': 1})
        assert agent.consumer is None