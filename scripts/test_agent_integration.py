import numpy as np
import time
import structlog
from src.ml.reinforcement_learning.online_agent import OnlineRLAgent
from unittest.mock import MagicMock, patch

logger = structlog.get_logger()

def simulate_integration():
    """
    Simulate the end-to-end flow: Market Data -> Online Agent -> Trading Signal.
    """
    logger.info("simulating_integration_start")
    
    # 1. Mock the model and Kafka
    with patch('stable_baselines3.TD3.load') as mock_load:
        mock_model = MagicMock()
        # Mock prediction: 10 target positions
        mock_model.predict.return_value = (np.random.uniform(-1, 1, 10).astype(np.float32), None)
        mock_load.return_value = mock_model
        
        # 2. Initialize Agent with mock config
        kafka_config = {'bootstrap.servers': 'localhost:9092'}
        with patch('confluent_kafka.Consumer'), patch('confluent_kafka.Producer'):
            agent = OnlineRLAgent(
                model_path="mock_model.zip", 
                kafka_config=kafka_config
            )
            
            # 3. Simulate market data message
            market_data = {
                'prices': np.random.uniform(90, 110, 10).tolist(),
                'greeks': np.random.uniform(-1, 1, (10, 5)).tolist(),
                'indicators': np.random.uniform(0, 1, 20).tolist()
            }
            
            # 4. Measure inference latency
            start_time = time.time()
            target_positions = agent.process_market_data(market_data)
            latency_ms = (time.time() - start_time) * 1000
            
            logger.info("inference_completed", 
                        latency_ms=f"{latency_ms:.2f}ms", 
                        positions=target_positions.tolist())
            
            # 5. Verify results
            assert len(target_positions) == 10
            assert latency_ms < 50, f"Latency {latency_ms}ms exceeds 50ms requirement"
            
            # 6. Verify transition storage
            assert len(agent.experience_buffer) == 0 # First call doesn't store (needs next state)
            
            # Second call should store the first transition
            agent.process_market_data(market_data)
            assert len(agent.experience_buffer) == 1
            logger.info("transition_storage_verified")
            
            logger.info("integration_test_passed")
            return True

if __name__ == "__main__":
    try:
        success = simulate_integration()
        if success:
            print("Integration Simulation: PASSED")
            exit(0)
    except Exception as e:
        print(f"Integration Simulation: FAILED - {str(e)}")
        exit(1)
