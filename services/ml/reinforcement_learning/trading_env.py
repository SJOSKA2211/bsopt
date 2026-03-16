from collections.abc import Callable
from typing import Any, cast

import gymnasium as gym
import numpy as np
import structlog
from gymnasium import spaces

from .kernels import _fused_state_kernel, _trading_step_kernel

logger = structlog.get_logger()


class TradingEnvironment(
    gym.Env[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]
):  # type: ignore
    """
    High-performance Trading Environment.
    FUSED: Uses Numba silicon kernels for zero-allocation state and reward logic.
    """

    def __init__(
        self,
        data_provider: Any = None,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.0001,
        window_size: int = 5,
    ) -> None:
        super().__init__()
        self.data_provider = data_provider
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        # Action space: target weights for 10 assets
        self.action_space: spaces.Box = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Observation space: (window_size, 128) for Transformer usage
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 128), dtype=np.float32
        )

        # Silicon Buffers
        self._window_buffer: np.ndarray[Any, np.dtype[np.float32]] = np.zeros(
            (window_size, 128), dtype=np.float32
        )

        # State variables
        self.balance: float = initial_balance
        self.positions: np.ndarray[Any, np.dtype[np.float32]] = np.zeros(10, dtype=np.float32)
        self.current_step: int = 0
        self.portfolio_values: list[float] = [initial_balance]
        self.market_data: dict[str, Any] = {}

        self.reset()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], dict[str, Any]]:
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.positions = np.zeros(10, dtype=np.float32)
        self.current_step = 0
        self.portfolio_values = [self.initial_balance]
        self._window_buffer.fill(0)

        if self.data_provider:
            self.market_data = cast(dict[str, Any], self.data_provider.get_data_at_step(0))
        else:
            self.market_data = self._get_dummy_data()

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray[Any, np.dtype[np.float32]]:
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

        # FUSION: Execute JIT kernel (Now using pre-allocated window buffer)
        f_state = cast(Callable[..., np.ndarray[Any, np.dtype[np.float32]]], _fused_state_kernel)
        return f_state(
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

    def step(
        self, action: np.ndarray[Any, np.dtype[np.float32]]
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment using fused machine-code kernel."""
        # 1. Clip and Prepare Input
        # Use cast to Any to access .low and .high on Space
        action_space = cast(Any, self.action_space)
        action_clipped = np.clip(action, action_space.low, action_space.high).astype(np.float32)
        prices = np.ascontiguousarray(
            self.market_data.get("prices", np.zeros(10))[:10], dtype=np.float32
        )
        pos = np.ascontiguousarray(self.positions, dtype=np.float32)

        # 2. FUSION: Execute Step Kernel
        f_step = cast(
            Callable[..., tuple[np.ndarray[Any, np.dtype[np.float32]], float, float, float]],
            _trading_step_kernel,
        )
        new_pos, new_balance, new_val, reward = f_step(
            action_clipped,
            prices,
            pos,
            float(self.balance),
            float(self.transaction_cost),
            float(self.initial_balance),
        )

        # 3. Commit state
        self.positions = new_pos
        self.balance = new_balance
        self.portfolio_values.append(new_val)

        # 4. Advance time
        self.current_step += 1
        if self.data_provider and self.current_step < len(self.data_provider):
            self.market_data = cast(
                dict[str, Any], self.data_provider.get_data_at_step(self.current_step)
            )
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
            vol = float(np.std(rets))
            ret -= 0.1 * vol

        # Drawdown penalty
        dd = (self.initial_balance - val) / self.initial_balance
        if dd > 0.1:
            ret -= 0.05 * (dd - 0.1)

        return float(ret)

    def _get_dummy_data(self) -> dict[str, Any]:
        """Generate random data for fallback/tests"""
        return {
            "prices": np.random.uniform(90, 110, 10),
            "strikes": np.random.uniform(90, 110, 10),
            "greeks": np.random.uniform(-1, 1, (10, 5)),
            "indicators": np.random.uniform(0, 1, 20),
        }
