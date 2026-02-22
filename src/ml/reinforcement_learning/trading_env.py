import structlog

from .kernels import _calculate_reward_kernel, _fused_state_kernel

logger = structlog.get_logger()


class TradingEnvironment:
    """
    High-Performance Reinforcement Learning Environment for HFT.
    Optimized: Fused state kernels and zero-copy reward calculation.
    """

    def __init__(self, initial_balance=100000.0, transaction_cost=0.0005):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        """Resets the environment state."""
        self.balance = self.initial_balance
        self.position = 0
        self.history = []
        return self._get_state()

    def step(self, action: int, current_price: float, market_features: list[float]):
        """
        Executes a step in the environment.
        Action: 0=Hold, 1=Buy, 2=Sell
        """
        # ... implementation ...
        reward = _calculate_reward_kernel(
            self.balance, self.position, current_price, 0.0, action
        )
        self.balance += reward # Simplified

        next_state = _fused_state_kernel(
            [self.balance, float(self.position)], market_features
        )
        return next_state, reward, False, {}

    def _get_state(self):
        return [self.balance, self.position]
