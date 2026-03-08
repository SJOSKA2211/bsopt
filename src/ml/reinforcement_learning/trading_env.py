import gymnasium as gym
import numpy as np
import structlog
from gymnasium import spaces

from .kernels import _calculate_reward_kernel, _fused_state_kernel, _trading_step_kernel

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
        """Execute one step in the environment using fused machine-code kernel."""
        # 1. Clip and Prepare Input
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        prices = np.ascontiguousarray(self.market_data.get("prices", np.zeros(10))[:10], dtype=np.float32)
        pos = np.ascontiguousarray(self.positions, dtype=np.float32)

        # 2. 🔥 FUSION: Execute Step Kernel
        new_pos, new_balance, new_val, reward = _trading_step_kernel(
            action,
            prices,
            pos,
            float(self.balance),
            float(self.transaction_cost),
            float(self.initial_balance)
        )

        # 3. Commit state
        self.positions = new_pos
        self.balance = new_balance
        self.portfolio_values.append(new_val)

        # 4. Advance time
        self.current_step += 1
        if self.data_provider and self.current_step < len(self.data_provider):
            self.market_data = self.data_provider.get_data_at_step(self.current_step)
        else:
            self.market_data = self._get_dummy_data()

        terminated = bool(self.data_provider and self.current_step >= len(self.data_provider) - 1)
        truncated = bool(new_val <= self.initial_balance * 0.5)

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            {"portfolio_value": new_val, "step": self.current_step},
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
