
import os
import time
import structlog
import numpy as np
from src.shared.shm_mesh import OrderBuffer, ExecutionBuffer, SHM_ORDER_NAME
from src.shared.observability import tune_gc

logger = structlog.get_logger(__name__)

class OrderEngine:
    """
    The Nerve Impulse: High-Frequency Order Entry Gateway.
    Spins on the lock-free OrderBuffer and fires binary trades.
    Pinned to Core 7 for zero-jitter execution.
    """
    def __init__(self):
        tune_gc()
        self.orders = OrderBuffer(create=False)
        self.execs = ExecutionBuffer(create=False)
        self._last_head = 0
        self._order_id_counter = 1000

    def run(self, cpu_core: int = 7):
        """Hot loop: Zero-latency order processing."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("order_engine_pinned", core=cpu_core)
        except Exception as e:
            logger.error("order_engine_pinning_failed", error=str(e))

        logger.info("order_engine_spinning", shm=SHM_ORDER_NAME)
        
        while True:
            # Poll order head
            import struct
            current_head = struct.unpack("q", self.orders.buf[:8])[0]
            
            if current_head > self._last_head:
                # New order command!
                cmd = self.orders.view[self._last_head % 1000]
                symbol = cmd['symbol'].decode().strip('\x00')
                price = float(cmd['price'])
                qty = int(cmd['quantity'])
                side = int(cmd['side'])
                
                # 🛡️ SOLENYA SHIELD: Silicon Risk Check
                from src.trading.risk_kernels import _validate_order_kernel
                is_safe = _validate_order_kernel(price, qty, side)
                
                if is_safe:
                    # 🚀 BINARY FIRE: Mock high-performance gateway
                    order_id = self._order_id_counter
                    self._order_id_counter += 1
                    
                    logger.info("order_fired_silicon", 
                                id=order_id, symbol=symbol, 
                                side="BUY" if side > 0 else "SELL", 
                                price=price, qty=qty)
                    
                    # Simulate instantaneous fill
                    self.execs.write_exec(order_id, price, qty, 1)
                else:
                    logger.warning("risk_violation_veto", symbol=symbol, price=price, qty=qty)
                    # Write rejection status (0) to ExecutionBuffer
                    self.execs.write_exec(-1, price, qty, 0)
                
                self._last_head += 1
            else:
                # Hint the CPU
                os.sched_yield()

if __name__ == "__main__":
    engine = OrderEngine()
    engine.run()
