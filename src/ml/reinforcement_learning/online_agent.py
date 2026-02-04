from collections import deque
import numpy as np
import structlog
from typing import Dict, Optional, Any
import orjson

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

from src.shared.observability import tune_gc

class OnlineRLAgent:
    """
    Online Reinforcement Learning Agent for generating trading signals from real-time Kafka streams.
    Supports Transformer-based temporal state stacking.
    """
    def __init__(self, 
                 model_path: str, 
                 initial_balance: float = 100000, 
                 kafka_config: Optional[Dict] = None,
                 input_topic: str = "market-data",
                 output_topic: str = "trading-signals",
                 use_transformer: bool = False,
                 window_size: int = 16):
        # Optimize Garbage Collection for the long-running RL loop
        tune_gc()
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = np.zeros(10, dtype=np.float32)
        self.kafka_config = kafka_config or {}
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.use_transformer = use_transformer
        self.window_size = window_size
        
        # Temporal history for Transformer attention
        self.history = deque(maxlen=window_size)
        self._state_buf = np.zeros(100, dtype=np.float32)
        
        # Load the trained model
        if TD3 is not None:
            try:
                self.model = TD3.load(model_path)
                logger.info("model_loaded", model_path=model_path, transformer=use_transformer)
            except Exception as e:
                self.model = None
                logger.error("model_load_failed", model_path=model_path, error=str(e))
        else:
            self.model = None
            logger.warning("stable_baselines3_not_found")

        # Initialize Kafka client
        self.consumer = None
        self.producer = None
        if Consumer is not None and self.kafka_config:
            self._init_kafka()

        # Continuous Learning state
        self.last_state = None
        self.last_action = None
        self.buffer_limit = 1000
        self._buffer_idx = 0
        
        # Memory-efficient pre-allocated buffer
        obs_dim = 100 * window_size if use_transformer else 100
        self._obs_buffer = np.zeros((self.buffer_limit, obs_dim), dtype=np.float32)
        self._act_buffer = np.zeros((self.buffer_limit, 10), dtype=np.float32)
        self._rew_buffer = np.zeros(self.buffer_limit, dtype=np.float32)
        self._next_obs_buffer = np.zeros((self.buffer_limit, obs_dim), dtype=np.float32)

    def _init_kafka(self):
        """Initialize Kafka consumer and producer."""
        try:
            consumer_config = self.kafka_config.copy()
            consumer_config.setdefault('group.id', 'rl-trading-agent')
            
            self.consumer = Consumer(consumer_config)
            self.consumer.subscribe([self.input_topic])
            
            self.producer = Producer({
                'bootstrap.servers': self.kafka_config.get('bootstrap.servers', 'localhost:9092')
            })
            logger.info("kafka_initialized", input_topic=self.input_topic)
        except Exception as e:
            logger.error("kafka_init_failed", error=str(e))

    def _get_state_vector(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Construct the state vector. 
        If use_transformer is True, returns a flattened window of history.
        """
        # 1. Portfolio state (11 dimensions)
        self._state_buf[0] = self.balance / self.initial_balance
        self._state_buf[1:11] = self.positions[:10]
        
        # 2. Market prices (10 dimensions) - Log-Moneyness
        prices = np.array(market_data.get('prices', []))
        strikes = np.array(market_data.get('strikes', prices))
        
        n_prices = min(len(prices), 10)
        if n_prices > 0:
            log_moneyness = np.log(prices[:n_prices] / strikes[:n_prices])
            self._state_buf[11:11+n_prices] = log_moneyness
            
        # 3. Greeks (50 dimensions) - Scaled via tanh
        greeks_raw = market_data.get('greeks', [])
        greeks_flat = np.array(greeks_raw).flatten()
        n_greeks = min(len(greeks_flat), 50)
        self._state_buf[21:21+n_greeks] = np.tanh(greeks_flat[:n_greeks])
        
        # 4. Indicators (20 dimensions)
        indicators = market_data.get('indicators', [])
        n_ind = min(len(indicators), 20)
        self._state_buf[71:71+n_ind] = indicators[:n_ind]
            
        current_obs = self._state_buf.copy()
        
        if self.use_transformer:
            if len(self.history) == 0:
                # Warm up history with current observation
                for _ in range(self.window_size):
                    self.history.append(current_obs)
            else:
                self.history.append(current_obs)
            return np.array(self.history).flatten()
            
        return current_obs

    def process_market_data(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Process market data and return target positions."""
        if self.model is None:
            return np.zeros(10)
            
        try:
            state = self._get_state_vector(market_data)
            
            # Continuous learning hook
            if self.last_state is not None and self.last_action is not None:
                reward = self._calculate_reward(market_data)
                self._store_transition(self.last_state, self.last_action, reward, state)
            
            action, _ = self.model.predict(state, deterministic=True)
            
            self.last_state = state
            self.last_action = action
            
            return action
        except Exception as e:
            logger.error("inference_failed", error=str(e))
            return np.zeros(10)

    def _calculate_reward(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate risk-adjusted reward for online learning consistency.
        Uses PnL and volatility penalty.
        """
        prices = np.array(market_data.get('prices', np.zeros(10)))
        if len(prices) != 10:
            prices = np.pad(prices, (0, max(0, 10 - len(prices))))[:10]
            
        # PnL Calculation
        current_option_value = np.sum(self.positions * prices)
        current_portfolio_value = self.balance + current_option_value
        
        # Initializing tracking on first call
        if not hasattr(self, '_prev_portfolio_value'):
            self._prev_portfolio_value = current_portfolio_value
            self._history_values = [current_portfolio_value]
            return 0.0

        ret = (current_portfolio_value - self._prev_portfolio_value) / self._prev_portfolio_value
        self._history_values.append(current_portfolio_value)
        self._prev_portfolio_value = current_portfolio_value
        
        # Keep sliding window for volatility penalty
        if len(self._history_values) > 20:
            self._history_values.pop(0)
            
        if len(self._history_values) >= 5:
            hist = np.array(self._history_values)
            returns = np.diff(hist) / hist[:-1]
            vol_penalty = 0.5 * np.std(returns)
        else:
            vol_penalty = 0.0
            
        reward = ret - vol_penalty
        return float(reward)

    def _store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        """Store experience transition in pre-allocated NumPy arrays."""
        idx = self._buffer_idx % self.buffer_limit
        self._obs_buffer[idx] = state
        self._act_buffer[idx] = action
        self._rew_buffer[idx] = reward
        self._next_obs_buffer[idx] = next_state
        
        self._buffer_idx += 1
        if self._buffer_idx % self.buffer_limit == 0:
            self._flush_experience()

    def _flush_experience(self):
        """Flush pre-allocated experience buffer to persistent storage or Kafka."""
        logger.info("flushing_experience", size=self.buffer_limit)
        # In a real system, we would serialize the numpy arrays (e.g. via .tobytes())
        # and send to a training service or save to disk.
        # For now, we just reset the index to simulate circular buffer usage
        pass

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
                    market_data = orjson.loads(msg.value().decode('utf-8'))
                    
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
