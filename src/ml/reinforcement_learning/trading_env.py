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
        
        # Observation space: 100-dimensional vector
        # [balance, positions (10), prices (10), greeks (50), indicators (20), padding (9)]
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
        """Construct observation vector from current state"""
        # Portfolio state (11 dimensions)
        portfolio_state = np.concatenate([
            [self.balance / self.initial_balance],
            self.positions
        ])
        
        # Market prices (10 dimensions)
        prices = self.market_data.get('prices', np.zeros(10))
        if len(prices) != 10:
            prices = np.pad(prices, (0, 10 - len(prices)))[:10]
        
        # Greeks (50 dimensions - assume 5 greeks per option)
        greeks = self.market_data.get('greeks', np.zeros((10, 5))).flatten()
        if len(greeks) != 50:
            greeks = np.pad(greeks, (0, 50 - len(greeks)))[:50]
        
        # Technical indicators (20 dimensions)
        indicators = self.market_data.get('indicators', np.zeros(20))
        if len(indicators) != 20:
            indicators = np.pad(indicators, (0, 20 - len(indicators)))[:20]
        
        # Combine all features
        observation = np.concatenate([
            portfolio_state, 
            prices, 
            greeks, 
            indicators
        ])
        
        # Pad to exactly 100 dimensions if needed
        if len(observation) < 100:
            observation = np.pad(observation, (0, 100 - len(observation)))
        else:
            observation = observation[:100]
            
        return observation.astype(np.float32)

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