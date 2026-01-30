import numpy as np
import structlog
from typing import Dict, Optional, Any
import json

try:
    from stable_baselines3 import TD3
except ImportError:
    TD3 = None

try:
    from confluent_kafka import Consumer, Producer, KafkaError
except ImportError:
    Consumer = None
    Producer = None

logger = structlog.get_logger()

import orjson

class OnlineRLAgent:
    """
    Online Reinforcement Learning Agent for generating trading signals from real-time Kafka streams.
    """
    def __init__(self, 
                 model_path: str, 
                 initial_balance: float = 100000, 
                 kafka_config: Optional[Dict] = None,
                 input_topic: str = "market-data",
                 output_topic: str = "trading-signals"):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = np.zeros(10, dtype=np.float32)
        self.kafka_config = kafka_config or {}
        self.input_topic = input_topic
        self.output_topic = output_topic
        
        # Pre-allocate state vector for speed
        self._state_buf = np.zeros(100, dtype=np.float32)
        
        # Load the trained model
        if TD3 is not None:
            try:
                self.model = TD3.load(model_path)
                logger.info("model_loaded", model_path=model_path)
            except Exception as e:
                self.model = None
                logger.error("model_load_failed", model_path=model_path, error=str(e))
        else:
            self.model = None
            logger.warning("stable_baselines3_not_found", message="TD3 model not loaded.")

        # Initialize Kafka client
        self.consumer = None
        self.producer = None
        if Consumer is not None and self.kafka_config:
            self._init_kafka()

        # Continuous Learning state
        self.last_state = None
        self.last_action = None
        self.experience_buffer = []
        self.buffer_limit = 1000

    def _init_kafka(self):
        """Initialize Kafka consumer and producer."""
        try:
            consumer_config = self.kafka_config.copy()
            if 'group.id' not in consumer_config:
                consumer_config['group.id'] = 'rl-trading-agent'
            
            self.consumer = Consumer(consumer_config)
            self.consumer.subscribe([self.input_topic])
            
            producer_config = {
                'bootstrap.servers': self.kafka_config.get('bootstrap.servers', 'localhost:9092')
            }
            self.producer = Producer(producer_config)
            logger.info("kafka_initialized", input_topic=self.input_topic, output_topic=self.output_topic)
        except Exception as e:
            logger.error("kafka_init_failed", error=str(e))

    def _get_state_vector(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Construct the state vector matching the TrainingEnvironment's observation space.
        Observation space (100 dims):
        - [0:11] Portfolio state (normalized balance, 10 positions)
        - [11:21] Market prices (10 options)
        - [21:71] Greeks (10 options * 5 greeks)
        - [71:91] Indicators (20 indicators)
        - [91:100] Padding
        """
        # 1. Portfolio state (11 dimensions)
        self._state_buf[0] = self.balance / self.initial_balance
        self._state_buf[1:11] = self.positions[:10]
        
        # 2. Market prices (10 dimensions)
        prices = market_data.get('prices', [])
        n_prices = min(len(prices), 10)
        self._state_buf[11:11+n_prices] = prices[:n_prices]
        if n_prices < 10:
            self._state_buf[11+n_prices:21] = 0.0
        
        # 3. Greeks (50 dimensions)
        greeks_raw = market_data.get('greeks', [])
        greeks_flat = np.array(greeks_raw).flatten()
        n_greeks = min(len(greeks_flat), 50)
        self._state_buf[21:21+n_greeks] = greeks_flat[:n_greeks]
        if n_greeks < 50:
            self._state_buf[21+n_greeks:71] = 0.0
        
        # 4. Indicators (20 dimensions)
        indicators = market_data.get('indicators', [])
        n_ind = min(len(indicators), 20)
        self._state_buf[71:71+n_ind] = indicators[:n_ind]
        if n_ind < 20:
            self._state_buf[71+n_ind:91] = 0.0
            
        # 5. Padding (9 dimensions)
        self._state_buf[91:100] = 0.0
            
        return self._state_buf.copy()

    def process_market_data(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Process market data and return target positions."""
        if self.model is None:
            logger.error("inference_skipped", reason="Model not loaded")
            return np.zeros(10)
            
        try:
            state = self._get_state_vector(market_data)
            
            # Continuous learning: If we have a previous state/action, store the transition
            if self.last_state is not None and self.last_action is not None:
                reward = self._calculate_reward(market_data)
                self._store_transition(self.last_state, self.last_action, reward, state)
            
            action, _states = self.model.predict(state, deterministic=True)
            
            # Update last state/action for next iteration
            self.last_state = state
            self.last_action = action
            
            return action
        except Exception as e:
            logger.error("inference_failed", error=str(e))
            return np.zeros(10)

    def _calculate_reward(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate reward for the last action based on current market data.
        In production, this would be more complex (PnL of the positions).
        """
        # Simplified: PnL based on price change
        prices = np.array(market_data.get('prices', np.zeros(10)))
        if len(prices) != 10:
            prices = np.pad(prices, (0, max(0, 10 - len(prices))))[:10]
            
        # This is a mock reward calculation
        return float(np.sum(self.positions * prices) / self.balance)

    def _store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        """Store experience transition for later offline/asynchronous retraining."""
        transition = {
            'state': state.tolist(),
            'action': action.tolist(),
            'reward': reward,
            'next_state': next_state.tolist()
        }
        self.experience_buffer.append(transition)
        if len(self.experience_buffer) > self.buffer_limit:
            self._flush_experience()

    def _flush_experience(self):
        """Flush experience buffer to persistent storage or Kafka."""
        if not self.experience_buffer:
            return
            
        logger.info("flushing_experience", size=len(self.experience_buffer))
        # In a real system, we might push this to a 'training-data' Kafka topic or a DB
        # For now, we'll just clear it to avoid memory leaks in this simulation
        self.experience_buffer = []

    def run(self):
        """Main loop for online inference."""
        if not self.consumer:
            logger.error("run_failed", reason="Kafka consumer not initialized")
            return

        logger.info("agent_running")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error("kafka_error", error=msg.error())
                        break
                
                # Parse market data
                try:
                    market_data = json.loads(msg.value().decode('utf-8'))
                    
                    # Generate action (target positions)
                    target_positions = self.process_market_data(market_data)
                    
                    # Send signals
                    self._produce_signal(target_positions)
                    
                    # Update internal tracking (simulation mode)
                    self.positions = target_positions
                    
                except Exception as e:
                    logger.error("msg_process_failed", error=str(e))
                    
        finally:
            self.consumer.close()

    def _produce_signal(self, target_positions: np.ndarray):
        """Produce trading signal to Kafka."""
        if not self.producer:
            return
            
        try:
            signal = {
                'agent_id': 'rl-agent-td3',
                'target_positions': target_positions.tolist(),
                'timestamp': np.datetime64('now').astype(str)
            }
            self.producer.produce(
                self.output_topic, 
                value=orjson.dumps(signal)
            )
            self.producer.flush()
        except Exception as e:
            logger.error("produce_failed", error=str(e))
