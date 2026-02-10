
import os

import numpy as np
import structlog
import torch

from src.ml.reinforcement_learning.kernels import _calculate_reward_kernel, _fused_state_kernel
from src.shared.observability import tune_gc
from src.shared.shm_mesh import (
    SHM_NAME,
    ExecutionBuffer,
    OrderBuffer,
    SharedMemoryRingBuffer,
)

logger = structlog.get_logger()

class OnlineRLAgent:
    """
    Advanced Online RL Agent with Neural Plasticity.
    Bypasses SB3. Uses TorchScript for silicon-speed inference.
    Supports zero-downtime weight hot-swapping.
    """
    def __init__(self, 
                 model_path: str, 
                 initial_balance: float = 100000, 
                 window_size: int = 16):
        tune_gc()
        self.model_path = model_path.replace(".zip", ".pt")
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = np.zeros(10, dtype=np.float32)
        self.window_size = window_size
        
        #  SILICON BUFFERS
        self._window_buffer = np.zeros((window_size, 100), dtype=np.float32)
        self._window_idx = 0
        self._prev_portfolio_value = initial_balance
        
        self._mesh = SharedMemoryRingBuffer(create=False)
        self._orders = OrderBuffer(create=False)
        self._execs = ExecutionBuffer(create=False)
        self._last_head = 0
        
        self.brain = None
        self._last_brain_mtime = 0
        self.reload_brain()

    def reload_brain(self):
        """Hot-swap the silicon weights if the file has changed."""
        try:
            if not os.path.exists(self.model_path):
                return
            mtime = os.path.getmtime(self.model_path)
            if mtime > self._last_brain_mtime:
                # Load new brain to a temp variable
                new_brain = torch.jit.load(self.model_path)
                new_brain.eval()
                # Warmup
                _ = new_brain(torch.zeros((1, 100)), torch.zeros((2, 10), dtype=torch.long))
                self.brain = new_brain
                self._last_brain_mtime = mtime
                logger.info("silicon_brain_reloaded", path=self.model_path)
        except Exception as e:
            if self.brain is None:
                logger.error("initial_brain_load_failed", error=str(e))

    def run(self, cpu_core: int = 2):
        """Hot loop: Pinned, spinning, and plastic."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("agent_pinned_to_core", core=cpu_core)
        except Exception as e:
            logger.error("agent_pinning_failed", error=str(e))

        logger.info("agent_spinning_on_mesh", shm=SHM_NAME)
        
        loop_count = 0
        try:
            with torch.no_grad():
                while True:
                    # 1. Plasticity Check (Every 1000 ticks)
                    if loop_count % 1000 == 0:
                        self.reload_brain()
                    
                    # 2. Poll SHM Mesh
                    view, current_head = self._mesh.read_latest_view(self._last_head)
                    
                    if current_head > self._last_head:
                        latest_tick = view[-1]
                        
                        # Construct state features (JIT Fused)
                        from src.pricing.factory import PricingEngineFactory
                        from src.pricing.models import BSParameters
                        
                        current_price = float(latest_tick['price'])
                        prices = np.full(10, current_price, dtype=np.float32)
                        strikes = np.full(10, 100.0, dtype=np.float32)
                        
                        engine = PricingEngineFactory.get_engine("black_scholes")
                        params = BSParameters(S=current_price, K=100.0, T=0.1, sigma=0.2, r=0.05)
                        g_vals = engine.calculate_greeks(params)
                        greeks = np.zeros(50, dtype=np.float32)
                        greeks[:5] = [g_vals.delta, g_vals.gamma, g_vals.theta, g_vals.vega, g_vals.rho]
                        
                        state_vector = _fused_state_kernel(
                            float(self.balance), float(self.initial_balance),
                            self.positions, prices, strikes,
                            greeks, np.zeros(20, dtype=np.float32),
                            self._window_buffer, self._window_idx, self.window_size
                        )
                        
                        # 3. Inference (SILICON)
                        if self.brain:
                            x = torch.from_numpy(state_vector).unsqueeze(0)
                            edge_index = torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9],
                                                       [1,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8]], dtype=torch.long)
                            action = self.brain(x, edge_index).detach().numpy()[0]
                            self._execute_action(action, prices)
                            
                            # Reward Fusion
                            new_val, _ = _calculate_reward_kernel(
                                self.positions, prices, self._prev_portfolio_value, self.balance
                            )
                            self._prev_portfolio_value = new_val
                        
                        self._window_idx += 1
                        self._last_head = current_head
                    else:
                        os.sched_yield()
                    
                    loop_count += 1
                    
        except KeyboardInterrupt:
            logger.info("agent_stopped")
        finally:
            self._mesh.close()

    def _execute_action(self, action: np.ndarray, prices: np.ndarray):
        """Drop binary orders into the SHM Mesh."""
        portfolio_value = self._prev_portfolio_value
        target_units = (action * portfolio_value) / (prices + 1e-9)
        trades = target_units - self.positions
        
        for i in range(len(trades)):
            qty = int(abs(trades[i]))
            if qty > 0:
                side = 1 if trades[i] > 0 else -1
                self._orders.write_order(f"OPT_{i}", float(prices[i]), qty, side)
                self.positions[i] = target_units[i]
                self.balance -= float(trades[i] * prices[i])

if __name__ == "__main__":
    agent = OnlineRLAgent("models/latest_td3.zip")
    agent.run()
