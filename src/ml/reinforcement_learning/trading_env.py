import gymnasium as gym
import numpy as np
import structlog
from gymnasium import spaces

from .kernels import _calculate_reward_kernel, _fused_state_kernel

logger = structlog.get_logger()


class TradingEnvironment(gym.Env):
    """
    High-performance Trading Environment.
    FUSED: Uses Numba silicon kernels for zero-allocation state and reward logic.
    """

    def __init__(
        self,
        data_provider=None,
        initial_balance=100000.0,
        transaction_cost=0.0001,
        window_size=5,
    ):
        super().__init__()
        self.data_provider = data_provider
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        # Action space: target weights for 10 assets
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

        # Observation space: (window_size, 100) for Transformer usage
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 100), dtype=np.float32
        )

        # Silicon Buffers
        self._window_buffer = np.zeros((window_size, 100), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.positions = np.zeros(10, dtype=np.float32)
        self.current_step = 0
        self.portfolio_values = [self.initial_balance]
        self._window_buffer.fill(0)

        if self.data_provider:
            self.market_data = self.data_provider.get_data_at_step(0)
        else:
            self.market_data = self._get_dummy_data()

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Fused state construction via silicon kernel."""
        prices = self.market_data.get("prices", np.ones(10))
        strikes = self.market_data.get("strikes", np.ones(10) * 100.0)
        greeks = self.market_data.get("greeks", np.zeros(50)).ravel()
        indicators = self.market_data.get("indicators", np.zeros(20))

        # Ensure correct shapes for kernel
        p = np.ascontiguousarray(prices[:10], dtype=np.float32)
        k = np.ascontiguousarray(strikes[:10], dtype=np.float32)
        g = np.ascontiguousarray(greeks[:50], dtype=np.float32)
        ind = np.ascontiguousarray(indicators[:20], dtype=np.float32)
        pos = np.ascontiguousarray(self.positions, dtype=np.float32)

        # 🔥 FUSION: Execute JIT kernel
        return _fused_state_kernel(
            float(self.balance),
            float(self.initial_balance),
            pos,
            p,
            k,
            g,
            ind,
            self._window_buffer,
            self.current_step,
            self.window_size,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get market state for current step
        current_prices = self.market_data.get("prices", np.zeros(10))
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
        self.balance -= transaction_costs + asset_costs

        # Advance time
        self.current_step += 1
        if self.data_provider and self.current_step < len(self.data_provider):
            self.market_data = self.data_provider.get_data_at_step(self.current_step)
        else:
            self.market_data = self._get_dummy_data()

        # Portfolio valuation & Reward via Silicon Kernel
        new_prices = np.ascontiguousarray(
            self.market_data.get("prices", np.zeros(10))[:10], dtype=np.float32
        )
        prev_val = self.portfolio_values[-1]

        # 🔥 FUSION: Reward Kernel
        current_val, reward = _calculate_reward_kernel(
            self.positions, new_prices, float(prev_val), float(self.balance)
        )

        self.portfolio_values.append(current_val)

        terminated = bool(self.data_provider and self.current_step >= len(self.data_provider) - 1)
        truncated = bool(current_val <= self.initial_balance * 0.5)

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            {"portfolio_value": current_val, "step": self.current_step},
        )

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
            "prices": np.random.uniform(90, 110, 10),
            "strikes": np.random.uniform(90, 110, 10),
            "greeks": np.random.uniform(-1, 1, (10, 5)),
            "indicators": np.random.uniform(0, 1, 20),
        }
