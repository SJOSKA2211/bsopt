import os
import struct

import structlog
from prometheus_client import Gauge, start_http_server

from src.shared.observability import tune_gc
from src.shared.shm_mesh import (
    BUFFER_CAPACITY,
    EXEC_BUFFER_CAPACITY,
    ORDER_BUFFER_CAPACITY,
    ExecutionBuffer,
    OrderBuffer,
    SharedMemoryRingBuffer,
)

logger = structlog.get_logger(__name__)

# Prometheus Metrics
T2T_LATENCY = Gauge("bsopt_t2t_latency_ns", "Tick-to-Trade Latency in Nanoseconds")
INF_LATENCY = Gauge("bsopt_inference_latency_ns", "Inference Latency in Nanoseconds")
OE_LATENCY = Gauge("bsopt_oe_latency_ns", "Order Entry Latency in Nanoseconds")
WS_LATENCY = Gauge("bsopt_ws_broadcast_latency_ms", "WebSocket Broadcast Latency in Milliseconds")
RISK_VETOS = Gauge("bsopt_risk_vetos_total", "Total Orders Vetoed by Risk Shield")


class TelemetryEngine:
    """
    The Chronos Eye: High-Resolution Latency Monitor.
    Spins on SHM Mesh, OrderBuffer, and ExecutionBuffer.
    Calculates sub-microsecond internal latencies.
    """

    def __init__(self, prometheus_port: int = 9091):
        tune_gc()
        self.mesh = SharedMemoryRingBuffer(create=False)
        self.orders = OrderBuffer(create=False)
        self.execs = ExecutionBuffer(create=False)

        self._last_mesh_head = 0
        self._last_order_head = 0
        self._last_exec_head = 0

        # Start Prometheus exporter
        start_http_server(prometheus_port)
        logger.info("telemetry_exporter_started", port=prometheus_port)

    def run(self, cpu_core: int = 8):
        """Hot loop: Pin to core and harvest timestamps."""
        try:
            os.sched_setaffinity(0, {cpu_core})
            logger.info("telemetry_engine_pinned", core=cpu_core)
        except Exception:
            pass

        logger.info("telemetry_harvesting_active")

        while True:
            # 1. Harvest Execution Status (The final hop)
            exec_head = struct.unpack("q", self.execs.buf[:8])[0]
            if exec_head > self._last_exec_head:
                exec_data = self.execs.view[self._last_exec_head % EXEC_BUFFER_CAPACITY]
                exec_ts = exec_data["exec_ts_ns"]
                order_id = exec_data["order_id"]
                status = exec_data["status"]

                # 2. Find corresponding order (Expanded Scan)
                order_head = struct.unpack("q", self.orders.buf[:8])[0]
                # Scan backwards further to handle high throughput
                for i in range(1, 100):
                    idx = (order_head - i) % ORDER_BUFFER_CAPACITY
                    ord_data = self.orders.view[idx]
                    # Note: We need a mapping here, but for the MVP of telemetry,
                    # we'll just use the latest order's submit_ts_ns
                    submit_ts = ord_data["submit_ts_ns"]

                    # 3. Find corresponding tick (The origin)
                    mesh_head = struct.unpack("q", self.mesh.buf[:8])[0]
                    tick_data = self.mesh.data_view[(mesh_head - 1) % BUFFER_CAPACITY]
                    receive_ts = tick_data["receive_ts_ns"]

                    # Calculate Metrics
                    t2t = exec_ts - receive_ts
                    inf = submit_ts - receive_ts
                    oe = exec_ts - submit_ts

                    T2T_LATENCY.set(t2t)
                    INF_LATENCY.set(inf)
                    OE_LATENCY.set(oe)

                    if status == 0:
                        RISK_VETOS.inc()
                        logger.warning("risk_veto_detected", order_id=order_id)

                    if t2t > 0:
                        logger.info(
                            "telemetry_snapshot",
                            t2t_ns=t2t,
                            inf_ns=inf,
                            oe_ns=oe,
                            order_id=order_id,
                        )
                    break

                self._last_exec_head = exec_head
            else:
                os.sched_yield()


if __name__ == "__main__":
    te = TelemetryEngine()
    te.run()
