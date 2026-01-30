import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple
import structlog

logger = structlog.get_logger()

class TradingEnvironment(gym.Env):
    """
    OpenAI Gym environment for options trading with real market data.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_balance: float = 100000, transaction_cost: float = 0.001, data_provider=None, risk_free_rate: float = 0.03):
        super().__init__()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.data_provider = data_provider
        self.risk_free_rate = risk_free_rate
        
        # Pre-allocate observation buffer for speed
        self._obs_buf = np.zeros(100, dtype=np.float32)
        
        # Observation space: 100-dimensional vector
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(100,), 
            dtype=np.float32
        )
        
        # Action space: position sizes for 10 options (-1 to 1)
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(10,), 
            dtype=np.float32
        )
        
        self.reset_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.positions = np.zeros(10) # 10 option positions
        self.current_step = 0
        self.portfolio_values = [self.initial_balance]
        
        # Get initial market data
        if self.data_provider:
            self.market_data = self.data_provider.get_latest_data()
        else:
            self.market_data = self._get_dummy_data()
            
        self.reset_count += 1
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current state using pre-allocated buffer"""
        # 1. Portfolio state (11 dimensions)
        self._obs_buf[0] = self.balance / self.initial_balance
        self._obs_buf[1:11] = self.positions[:10]
        
        # 2. Market prices (10 dimensions)
        prices = self.market_data.get('prices', [])
        n_prices = min(len(prices), 10)
        self._obs_buf[11:11+n_prices] = prices[:n_prices]
        if n_prices < 10:
            self._obs_buf[11+n_prices:21] = 0.0
        
        # 3. Greeks (50 dimensions)
        greeks_raw = self.market_data.get('greeks', [])
        greeks_flat = np.array(greeks_raw).flatten()
        n_greeks = min(len(greeks_flat), 50)
        self._obs_buf[21:21+n_greeks] = greeks_flat[:n_greeks]
        if n_greeks < 50:
            self._obs_buf[21+n_greeks:71] = 0.0
        
        # 4. Technical indicators (20 dimensions)
        indicators = self.market_data.get('indicators', [])
        n_ind = min(len(indicators), 20)
        self._obs_buf[71:71+n_ind] = indicators[:n_ind]
        if n_ind < 20:
            self._obs_buf[71+n_ind:91] = 0.0
            
        # 5. Padding (9 dimensions)
        self._obs_buf[91:100] = 0.0
            
        return self._obs_buf.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        """
        # 0. Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 1. Execute trades (rebalance to target positions)
        trades = action - self.positions
        
        # 2. Calculate costs
        current_prices = self.market_data.get('prices', np.zeros(10))
        if len(current_prices) != 10:
            current_prices = np.pad(current_prices, (0, 10 - len(current_prices)))[:10]
            
        transaction_costs = np.sum(np.abs(trades) * current_prices * self.transaction_cost)
        asset_costs = np.sum(trades * current_prices)
        
        # 3. Update state
        self.positions = action
        self.balance -= (transaction_costs + asset_costs)
        
        # 4. Advance time
        self.current_step += 1
        if self.data_provider:
            self.market_data = self.data_provider.get_data_at_step(self.current_step)
        else:
            self.market_data = self._get_dummy_data()
            
        # 5. Calculate portfolio value
        current_prices = self.market_data.get('prices', np.zeros(10))
        if len(current_prices) != 10:
            current_prices = np.pad(current_prices, (0, 10 - len(current_prices)))[:10]
            
        option_values = np.sum(self.positions * current_prices)
        portfolio_value = self.balance + option_values
        self.portfolio_values.append(portfolio_value)
        
        # 6. Calculate reward (Sharpe-like risk adjusted return)
        reward = self._calculate_reward(portfolio_value)
        
        # 7. Check termination
        terminated = False
        if self.data_provider and self.current_step >= len(self.data_provider) - 1:
            terminated = True
        elif not self.data_provider and self.current_step >= 100:
            terminated = True
            
        # 8. Check truncation (Drawdown limit)
        truncated = bool(portfolio_value <= self.initial_balance * 0.5)
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'step': self.current_step
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_reward(self, current_portfolio_value: float) -> float:
        """
        Multi-objective reward function incorporating risk-adjusted returns.
        """
        # 1. Percentage return
        prev_value = self.portfolio_values[-2]
        ret = (current_portfolio_value - prev_value) / prev_value
        
        # 2. Volatility penalty (risk adjustment)
        # Use recent portfolio returns to calculate volatility
        if len(self.portfolio_values) > 5:
            recent_values = np.array(self.portfolio_values[-5:])
            recent_returns = np.diff(recent_values) / recent_values[:-1]
            volatility = np.std(recent_returns)
            # Penalty proportional to volatility
            vol_penalty = 0.1 * volatility
        else:
            vol_penalty = 0
            
        reward = ret - vol_penalty
        
        # 3. Drawdown penalty
        if current_portfolio_value < self.initial_balance * 0.8:
            reward -= 0.01 * (1.0 - current_portfolio_value / (self.initial_balance * 0.8))
            
        return float(reward)

    def _get_dummy_data(self) -> Dict:
        """Generate random data for fallback/tests"""
        return {
            'prices': np.random.uniform(90, 110, 10),
            'greeks': np.random.uniform(-1, 1, (10, 5)),
            'indicators': np.random.uniform(0, 1, 20)
        }