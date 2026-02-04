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

    def __init__(self, initial_balance: float = 100000, transaction_cost: float = 0.001, data_provider=None, risk_free_rate: float = 0.03, window_size: int = 16):
        super().__init__()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.data_provider = data_provider
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        
        # Pre-allocate sequence buffer: [window_size, 100]
        self._obs_buf = np.zeros((window_size, 100), dtype=np.float32)
        
        # Observation space: 2D matrix [window, features]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size, 100), 
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
        
        # Reset history
        self._obs_buf.fill(0.0)
        
        # Get initial market data
        if self.data_provider:
            self.market_data = self.data_provider.get_latest_data()
        else:
            self.market_data = self._get_dummy_data()
            
        # Prime the history with the initial state
        initial_obs = self._construct_single_obs()
        for i in range(self.window_size):
            self._obs_buf[i] = initial_obs
            
        self.reset_count += 1
        return self._obs_buf.copy(), {}

    def _construct_single_obs(self) -> np.ndarray:
        """Constructs a single 100-dim observation vector."""
        vec = np.zeros(100, dtype=np.float32)
        # 1. Portfolio state (11 dimensions)
        vec[0] = self.balance / self.initial_balance
        vec[1:11] = self.positions[:10]
        
        # 2. Market prices (10 dimensions)
        prices = np.array(self.market_data.get('prices', []))
        strikes = np.array(self.market_data.get('strikes', prices))
        n_prices = min(len(prices), 10)
        if n_prices > 0:
            vec[11:11+n_prices] = np.log(prices[:n_prices] / strikes[:n_prices])
        
        # 3. Greeks (50 dimensions)
        greeks_raw = self.market_data.get('greeks', [])
        greeks_flat = np.array(greeks_raw).flatten()
        n_greeks = min(len(greeks_flat), 50)
        vec[21:21+n_greeks] = np.tanh(greeks_flat[:n_greeks])
        
        # 4. Indicators (20 dimensions)
        indicators = self.market_data.get('indicators', [])
        n_ind = min(len(indicators), 20)
        vec[71:71+n_ind] = indicators[:n_ind]
            
        return vec

    def _get_observation(self) -> np.ndarray:
        """Update sliding window and return full history."""
        # Shift history
        self._obs_buf = np.roll(self._obs_buf, -1, axis=0)
        # Append latest
        self._obs_buf[-1] = self._construct_single_obs()
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
        current_prices = np.array(self.market_data.get('prices', np.zeros(10)))
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
        current_prices = np.array(self.market_data.get('prices', np.zeros(10)))
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
        Improved stability using larger window and downside risk focus.
        """
        # 1. Percentage return
        prev_value = self.portfolio_values[-2]
        ret = (current_portfolio_value - prev_value) / prev_value
        
        # 2. Volatility penalty (risk adjustment) - Increased window to 20 for stability
        if len(self.portfolio_values) > 20:
            recent_values = np.array(self.portfolio_values[-20:])
            recent_returns = np.diff(recent_values) / recent_values[:-1]
            volatility = np.std(recent_returns)
            
            # Sortino-style penalty: focus more on negative volatility if needed
            # For now, stable std dev is better for TD3 gradients
            vol_penalty = 0.5 * volatility # Increased weight on risk
        else:
            vol_penalty = 0
            
        reward = ret - vol_penalty
        
        # 3. Progressive Drawdown penalty
        drawdown = (self.initial_balance - current_portfolio_value) / self.initial_balance
        if drawdown > 0.1: # 10% threshold
            reward -= 0.1 * (drawdown - 0.1)
            
        return float(reward)

    def _get_dummy_data(self) -> Dict:
        """Generate random data for fallback/tests"""
        return {
            'prices': np.random.uniform(90, 110, 10),
            'greeks': np.random.uniform(-1, 1, (10, 5)),
            'indicators': np.random.uniform(0, 1, 20)
        }