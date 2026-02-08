
import gymnasium as gym
import numpy as np
import structlog
from gymnasium import spaces

logger = structlog.get_logger()

class TradingEnvironment(gym.Env):
    def __init__(self, data_provider=None, initial_balance=100000.0, transaction_cost=0.0001):
        super().__init__()
        self.data_provider = data_provider
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Action space: target weights for 10 assets
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        
        # Observation space: 100 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)
        
        # Pre-allocate observation buffer for zero-allocation construction
        self._obs_buffer = np.zeros(100, dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.positions = np.zeros(10, dtype=np.float32)
        self.current_step = 0
        self.portfolio_values = [self.initial_balance]
        
        if self.data_provider:
            self.market_data = self.data_provider.get_data_at_step(0)
        else:
            self.market_data = self._get_dummy_data()
            
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Constructs observation from market data and portfolio state."""
        # 1. Portfolio state (11 dimensions)
        self._obs_buffer[0] = self.balance / self.initial_balance
        self._obs_buffer[1:11] = self.positions
        
        # 2. Market prices (10 dimensions)
        prices = self.market_data.get('prices')
        # Fix: Use a fixed reference if strikes are missing to avoid log(1)=0
        strikes = self.market_data.get('strikes', np.ones_like(prices) * 100.0)
        
        # Efficient validation and log-return
        p = np.maximum(prices, 1e-6)
        k = np.maximum(strikes, 1e-6)
        n_prices = min(len(p), 10)
        self._obs_buffer[11:11+n_prices] = np.log(p[:n_prices] / k[:n_prices])

        
        # 3. Greeks (50 dimensions)
        greeks = self.market_data.get('greeks')
        if greeks is not None:
            greeks_flat = greeks.ravel()
            n_greeks = min(len(greeks_flat), 50)
            self._obs_buffer[21:21+n_greeks] = np.tanh(greeks_flat[:n_greeks])
        
        # 4. Indicators (20 dimensions)
        indicators = self.market_data.get('indicators')
        if indicators is not None:
            n_ind = min(len(indicators), 20)
            self._obs_buffer[71:71+n_ind] = indicators[:n_ind]
            
        # Global sanity check
        if not np.all(np.isfinite(self._obs_buffer)):
            self._obs_buffer[~np.isfinite(self._obs_buffer)] = 0
            
        return self._obs_buffer.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Get market state for current step
        current_prices = self.market_data.get('prices')
        if current_prices is None or len(current_prices) < 10:
             current_prices = np.zeros(10) # Fallback
        
        p_safe = np.maximum(current_prices[:10], 1e-6)
        
        # Calculate current portfolio value for weight conversion
        portfolio_value = self.balance + np.sum(self.positions * p_safe)
        
        # Convert weight actions to target units
        target_units = (action * portfolio_value) / (p_safe + 1e-9)
        trades = target_units - self.positions
        
        # Rebalancing costs
        transaction_costs = np.sum(np.abs(trades) * p_safe * self.transaction_cost)
        asset_costs = np.sum(trades * p_safe)
        
        # Update state
        self.positions = target_units
        self.balance -= (transaction_costs + asset_costs)

        
        if self.balance < -1e6: # Sanity limit
            return self._get_observation(), -10.0, True, True, {}
            
        # Advance time
        self.current_step += 1
        if self.data_provider and self.current_step < len(self.data_provider):
            self.market_data = self.data_provider.get_data_at_step(self.current_step)
        else:
            self.market_data = self._get_dummy_data()
            
        # Portfolio valuation
        new_prices = self.market_data.get("prices")
        if new_prices is None:
            new_prices = np.zeros(10)
        
        portfolio_value = self.balance + np.sum(self.positions * new_prices[:10])
        self.portfolio_values.append(portfolio_value)
        
        # Reward & Limits
        reward = self._calculate_reward(portfolio_value)
        
        terminated = bool(self.data_provider and self.current_step >= len(self.data_provider) - 1)
        truncated = bool(portfolio_value <= self.initial_balance * 0.5)
        
        return self._get_observation(), reward, terminated, truncated, {
            'portfolio_value': portfolio_value,
            'step': self.current_step
        }

    def _calculate_reward(self, val: float) -> float:
        prev = self.portfolio_values[-2]
        ret = (val - prev) / (prev + 1e-9)
        
        # Volatility penalty (sharpe-like)
        if len(self.portfolio_values) > 10:
            window = np.array(self.portfolio_values[-10:])
            rets = np.diff(window) / (window[:-1] + 1e-9)
            vol = np.std(rets)
            ret -= 0.1 * vol
            
        # Drawdown penalty
        dd = (self.initial_balance - val) / self.initial_balance
        if dd > 0.1:
            ret -= 0.05 * (dd - 0.1)
            
        return float(ret)

    def _get_dummy_data(self) -> dict:
        """Generate random data for fallback/tests"""
        return {
            'prices': np.random.uniform(90, 110, 10),
            'strikes': np.random.uniform(90, 110, 10),
            'greeks': np.random.uniform(-1, 1, (10, 5)),
            'indicators': np.random.uniform(0, 1, 20)
        }