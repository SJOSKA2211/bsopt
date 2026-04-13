"""
Query Variance Report Script - BS-OPT Tooling
Analyzes execution jitter for mission-critical risk and pricing queries.
"""

import time

import numpy as np
import structlog
from sqlalchemy import text

from src.database import get_engine

logger = structlog.get_logger(__name__)


def run_variance_audit(iterations: int = 100):
    engine = get_engine()
    print("  BSOpt Query Jitter Audit (High-Performance)")
    print("--")

    queries = {
        "latest_options_prices": "SELECT * FROM options_prices ORDER BY time DESC LIMIT 100",
        "risk_aggregation": "SELECT symbol, sum(delta) as total_delta FROM options_prices GROUP BY symbol",
        "portfolio_pnl": "SELECT portfolio_id, sum(realized_pnl) FROM positions GROUP BY portfolio_id",
    }

    for name, sql in queries.items():
        latencies = []
        print(f"Auditing [{name}]...")

        with engine.connect() as conn:
            # Warm up
            conn.execute(text(sql)).fetchall()

            for _ in range(iterations):
                start = time.perf_counter()
                conn.execute(text(sql)).fetchall()
                latencies.append((time.perf_counter() - start) * 1000)

        lats = np.array(latencies)
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        p99 = np.percentile(lats, 99)
        std = np.std(lats)

        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")
        print(f"  StdDev: {std:.4f}ms")

        if p99 > 10.0:  # Arbitrary threshold for "high jitter"
            print(f"   ALERT: High P99 jitter detected for {name}!")
        else:
            print("   Stability: EXCELLENT")
        print("")


if __name__ == "__main__":
    run_variance_audit()