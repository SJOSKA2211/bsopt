import time
from typing import Any

import numpy as np
import structlog

from src.ml.reinforcement_learning.kernels import _calculate_reward_kernel, _fused_state_kernel
from src.shared.observability import tune_gc
from src.shared.shm_mesh import SHM_NAME, TICK_DTYPE, SharedMemoryRingBuffer

try:
    from stable_baselines3 import TD3
except ImportError:
    TD3 = None

logger = structlog.get_logger()

class OnlineRLAgent:
    """
    God-Mode Online RL Agent.
    Bypasses Kafka. Spins on the lock-free SHM Mesh.
    Uses fused JIT kernels for state construction.
    """
    def __init__(self, 
                 model_path: str, 
                 initial_balance: float = 100000, 
                 window_size: int = 16):
        tune_gc()
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = np.zeros(10, dtype=np.float32)
        self.window_size = window_size
        
        # 🚀 SILICON BUFFERS: Pre-allocated for zero-allocation loops
        self._window_buffer = np.zeros((window_size, 100), dtype=np.float32)
        self._window_idx = 0
        self._prev_portfolio_value = initial_balance
        
        # Initialize Mesh Reader
        self._mesh = SharedMemoryRingBuffer(create=False)
        self._last_head = 0
        
        # Load the trained model
        if TD3 is not None:
            try:
                self.model = TD3.load(model_path)
                logger.info("model_loaded_to_silicon", model_path=model_path)
            except Exception as e:
                self.model = None
                logger.error("model_load_failed", error=str(e))
        else:
            self.model = None

    def run(self, cpu_core: int = 2):
        """
        Hot loop: Pinned to core, spinning on SHM Mesh. 
        Zero-latency inference path.
        """
        # 🚀 SILICON LOCKDOWN: Pin to core
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("agent_pinned_to_core", core=cpu_core)
        except Exception as e:
            logger.error("agent_pinning_failed", error=str(e))

        logger.info("agent_spinning_on_mesh", shm=SHM_NAME)
        
        try:
            while True:
                # 🚀 SPIN LOCK: Poll the SHM head index
                view, current_head = self._mesh.read_latest_view(self._last_head)
                
                if current_head > self._last_head:
                    # New ticks detected!
                    latest_tick = view[-1]
                    
                    # 1. Update State & Generate State Vector (Fused JIT)
                    prices = np.full(10, latest_tick['price'], dtype=np.float32)
                    strikes = np.full(10, 100.0, dtype=np.float32)
                    greeks = np.zeros(50, dtype=np.float32)
                    indicators = np.zeros(20, dtype=np.float32)
                    
                    state_vector = _fused_state_kernel(
                        self.balance, self.initial_balance,
                        self.positions, prices, strikes,
                        greeks, indicators,
                        self._window_buffer, self._window_idx, self.window_size
                    )
                    
                    # 2. Inference
                    if self.model:
                        action, _ = self.model.predict(state_vector, deterministic=True)
                        
                        # 3. Action Execution (Simulated)
                        self._execute_action(action, prices)
                        
                        # 4. Reward & Portfolio Tracking (Fused JIT)
                        new_val, ret = _calculate_reward_kernel(
                            self.positions, prices, self._prev_portfolio_value, self.balance
                        )
                        self._prev_portfolio_value = new_val
                    
                    self._window_idx += 1
                    self._last_head = current_head
                else:
                    # 🚀 ZERO-LATENCY YIELD: Hints the CPU but stays in the spin
                    os.sched_yield()
                    
        except KeyboardInterrupt:
            logger.info("agent_stopped")
        finally:
            self._mesh.close()

    def _execute_action(self, action: np.ndarray, prices: np.ndarray):
        """Execute trades against current balance."""
        # Convert weights to units
        portfolio_value = self._prev_portfolio_value
        target_units = (action * portfolio_value) / (prices + 1e-9)
        trades = target_units - self.positions
        cost = np.sum(trades * prices)
        
        self.balance -= cost
        self.positions = target_units

if __name__ == "__main__":
    # In prod, this would be started by the orchestrator
    agent = OnlineRLAgent("models/latest_td3.zip")
    agent.run()
