import os
import time
from typing import Any

import numpy as np
import structlog

from src.ml.reinforcement_learning.kernels import _calculate_reward_kernel, _fused_state_kernel
from src.shared.observability import tune_gc
from src.shared.shm_mesh import SHM_NAME, TICK_DTYPE, SharedMemoryRingBuffer, OrderBuffer, ExecutionBuffer

try:
    from stable_baselines3 import TD3
except ImportError:
    TD3 = None

logger = structlog.get_logger()

import torch

class OnlineRLAgent:
    """
    God-Mode Online RL Agent.
    Bypasses SB3. Uses TorchScript for silicon-speed inference.
    Spins on the lock-free SHM Mesh.
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
        
        # Initialize Mesh Reader & Order Nervous System
        self._mesh = SharedMemoryRingBuffer(create=False)
        self._orders = OrderBuffer(create=False)
        self._execs = ExecutionBuffer(create=False)
        self._last_head = 0
        
        # Load the Silicon Brain (TorchScript)
        try:
            # We expect a .pt file exported from the training script
            self.brain = torch.jit.load(model_path.replace(".zip", ".pt"))
            self.brain.eval()
            # 🚀 WARMUP: Prime the JIT compiler
            _ = self.brain(torch.zeros((1, 100)), torch.zeros((2, 10), dtype=torch.long))
            logger.info("silicon_brain_loaded", path=model_path)
        except Exception as e:
            self.brain = None
            logger.error("silicon_brain_load_failed", error=str(e))

    def run(self, cpu_core: int = 2):
        # ... (same as before)
        
        try:
            with torch.no_grad():
                while True:
                    # 🚀 SPIN LOCK
                    view, current_head = self._mesh.read_latest_view(self._last_head)
                    
                    if current_head > self._last_head:
                        # ... (state vector construction)
                        
                        # 2. Inference (SILICON PATH)
                        if self.brain:
                            # Convert state_vector to torch tensor (Zero-copy if possible)
                            x = torch.from_numpy(state_vector).unsqueeze(0)
                            # Edge index for 10 nodes (OPT_0 to OPT_9)
                            edge_index = torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9],
                                                       [1,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8]], dtype=torch.long)
                            
                            # Fire the Silicon Brain!
                            action = self.brain(x, edge_index).numpy()[0]
                            
                            # 3. Action Execution
                            self._execute_action(action, prices)
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
                        from src.pricing.factory import PricingEngineFactory
                        from src.pricing.models import BSParameters
                        
                        current_price = float(latest_tick['price'])
                        prices = np.full(10, current_price, dtype=np.float32)
                        strikes = np.full(10, 100.0, dtype=np.float32)
                        
                        # 🔥 FUSION: Real-time Greek Calculation
                        engine = PricingEngineFactory.get_engine("black_scholes")
                        params = BSParameters(S=current_price, K=100.0, T=0.1, sigma=0.2, r=0.05)
                        g_vals = engine.calculate_greeks(params)
                        greeks = np.zeros(50, dtype=np.float32)
                        greeks[:5] = [g_vals.delta, g_vals.gamma, g_vals.theta, g_vals.vega, g_vals.rho]
                        
                        indicators = np.zeros(20, dtype=np.float32)
                        
                        state_vector = _fused_state_kernel(
                            float(self.balance), float(self.initial_balance),
                            self.positions, prices, strikes,
                            greeks, indicators,
                            self._window_buffer, self._window_idx, self.window_size
                        )
                        
                        # 2. Inference (SILICON BRAIN)
                        if self.brain:
                            # Convert to torch tensor
                            x = torch.from_numpy(state_vector).unsqueeze(0)
                            # Static edge index for 10 nodes
                            edge_index = torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9],
                                                       [1,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8]], dtype=torch.long)
                            
                            # Fire inference!
                            action = self.brain(x, edge_index).detach().numpy()[0]
                            
                            # 3. Action Execution
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
        """Drop binary orders into the SHM Mesh."""
        # Convert weights to units
        portfolio_value = self._prev_portfolio_value
        target_units = (action * portfolio_value) / (prices + 1e-9)
        trades = target_units - self.positions
        
        # Drop orders for each non-trivial trade
        for i in range(len(trades)):
            qty = int(abs(trades[i]))
            if qty > 0:
                side = 1 if trades[i] > 0 else -1
                # Write to lock-free OrderBuffer (Direct silicon path to Core 7)
                self._orders.write_order(f"OPT_{i}", float(prices[i]), qty, side)
                
                # Optimistic local update
                self.positions[i] = target_units[i]
                self.balance -= float(trades[i] * prices[i])

if __name__ == "__main__":
    # In prod, this would be started by the orchestrator
    agent = OnlineRLAgent("models/latest_td3.zip")
    agent.run()
