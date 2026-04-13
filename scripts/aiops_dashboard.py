import asyncio
import os
from datetime import datetime

from src.shared.config import settings

# ANSI Colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_status(status: str) -> str:
    color = GREEN if status == "healthy" else (YELLOW if status == "degraded" else RED)
    return f"{color}{BOLD}{status.upper()}{RESET}"


def print_section(title: str, color=CYAN):
    print(f"\n{color}{BOLD}{'=' * 60}{RESET}")
    print(f"{color}{BOLD}  {title}{RESET}")
    print(f"{color}{BOLD}{'=' * 60}{RESET}")


async def run_dashboard():
    # Setup
    os.environ["BSOPT_ALLOW_WEAK_SECRETS"] = "True"
    
    try:
        from src.ml.aiops.health_reporter import HealthReporter
    except ImportError as e:
        print(f"\n{RED} ERROR: Could not import HealthReporter (likely missing ML dependencies or CUDA error): {e}{RESET}")
        return

    reporter = HealthReporter(prometheus_url=os.environ.get("PROMETHEUS_URL", settings.PROMETHEUS_URL))

    # In a real environment, we'd poll the running engine.
    # Here we simulate a snapshort of the Manifold state.
    print_section("AIOPS MANIFOLD TERMINAL DASHBOARD", color=MAGENTA)
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Fetch actual health (will fail if DB/Redis not reachable, but we show the degraded state)
        report = await reporter.get_health_report()

        print(f" Overall Status: {format_status(report.status)}")

        # 1. Infrastructure Layer
        print_section("INFRASTRUCTURE LAYER")

        # Postgres
        pg = report.postgres
        pg_icon = "" if pg.connected else ""
        print(f"{pg_icon}  POSTGRES/TIMESCALE:")
        print(f"   - Connected: {pg.connected}")
        print(f"   - Active Connections: {pg.active_connections}")
        print(f"   - Hypertables: {pg.hypertables}")
        print(f"   - Compression Ratio: {pg.compression_ratio}:1")

        # RabbitMQ
        rmq = report.rabbitmq
        rmq_icon = "" if rmq.connected else ""
        print(f"\n{rmq_icon}  RABBITMQ:")
        print(f"   - Connected: {rmq.connected}")
        print(f"   - Total Queues: {len(rmq.queue_depths)}")
        if rmq.queue_depths:
            # Show first 3 queues
            for idx, (q_name, depth) in enumerate(list(rmq.queue_depths.items())[:3]):
                consumers = rmq.consumer_counts.get(q_name, 0)
                print(f"     └─ {q_name}: {depth} msgs, {consumers} consumers")

        # Redis
        rd = report.redis
        rd_icon = "" if rd.connected else ""
        print(f"\n{rd_icon}  REDIS:")
        print(f"   - Connected: {rd.connected}")
        print(f"   - Memory Usage: {rd.memory_usage_bytes / (1024 * 1024):.2f} MB")
        print(f"   - Total Keys: {rd.total_keys}")

        # API (REST Gateway)
        api = report.api
        api_icon = "" if api.reachable else ""
        print(f"\n{api_icon}  REST API GATEWAY:")
        print(f"   - Reachable: {api.reachable}")
        print(f"   - P95 Latency: {api.p95_latency:.4f}s")
        print(f"   - 5xx Error Rate: {api.error_rate_5xx:.4f}")
        print(f"   - Request Count: {api.request_count}")

        # Auth (Security Gateway)
        auth = report.auth
        auth_icon = "" if auth.reachable else ""
        print(f"\n{auth_icon}  AUTHENTICATION GATEWAY:")
        print(f"   - Reachable: {auth.reachable}")
        print(f"   - P95 Latency: {auth.p95_latency:.4f}s")
        print(f"   - Auth Success Rate: {auth.auth_success_rate:.2%}")
        print(f"   - Active Tokens: {auth.active_tokens}")

        # Ingestion (Data Pipeline)
        ing = report.ingestion
        ing_icon = "" if ing.reachable else "️"
        print(f"\n{ing_icon}  DATA INGESTION LAYER:")
        print(f"   - Heartbeat Age: {ing.heartbeat_age:.1f}s")
        print(f"   - Throughput (TPS): {ing.ticks_per_second:.2f}")
        print(f"   - Tick Rejection Rate: {ing.rejection_rate:.2%}")

        # Portfolio (Risk & Exposure)
        port = report.portfolio
        port_icon = "" if port.reachable else "️"
        print(f"\n{port_icon}  PORTFOLIO & RISK LAYER:")
        print(f"   - Reachable: {port.reachable}")
        print(f"   - Position Count: {port.positions_count}")
        print(f"   - Net Delta: {port.net_delta:+.2f}")
        print(f"   - Total Vega: {port.total_vega:.2f}")
        print(f"   - Total Gamma: {port.total_gamma:.4f}")

        # Quant (Math Kernel)
        quant = report.quant
        quant_icon = "🧮" if quant.reachable else "️"
        print(f"\n{quant_icon}  QUANT & MATH KERNEL LAYER:")
        print(f"   - Reachable: {quant.reachable}")
        print(f"   - Avg Latency: {quant.avg_latency_ms:.2f}ms")
        print(f"   - Throughput: {quant.requests_per_sec:.2f} req/s")
        print(f"   - Error Rate: {quant.error_rate:.2%}")

        # ML Inference (Neural Pricing)
        inf = report.inference
        inf_icon = "🧠" if inf.reachable and inf.model_loaded else "️"
        model_status = "LOADED " if inf.model_loaded else "NOT LOADED "
        print(f"\n{inf_icon}  ML INFERENCE LAYER:")
        print(f"   - Reachable: {inf.reachable}")
        print(f"   - Model State: {model_status}")
        print(f"   - Avg Latency: {inf.avg_latency_ms:.2f}ms")
        print(f"   - Throughput: {inf.requests_per_sec:.2f} req/s")

        # Worker Cluster (Celery)
        wrk = report.workers
        wrk_icon = "️ " if wrk.reachable and wrk.active_workers > 0 else ""
        print(f"\n{wrk_icon}  WORKER CLUSTER (CELERY):")
        print(f"   - Reachable: {wrk.reachable}")
        print(f"   - Active Workers: {wrk.active_workers}")
        print(f"   - Avg Task Latency: {wrk.avg_task_latency_ms:.2f}ms")
        if wrk.queue_backlog:
            print("   - Queue Backlog:")
            for q, depth in wrk.queue_backlog.items():
                print(f"     └─ {q}: {depth} msgs")

        # Ray Cluster (Distributed Actors)
        ray_s = report.ray
        ray_icon = "️ " if ray_s.reachable and ray_s.nodes_alive > 0 else "️"
        print(f"\n{ray_icon}  RAY CLUSTER (DISTRIBUTED):")
        print(f"   - Reachable: {ray_s.reachable}")
        print(f"   - Nodes Alive: {ray_s.nodes_alive}")
        print(f"   - Actor Count: {ray_s.worker_count}")

        # 2. Autonomous Oversight
        print_section("AUTONOMOUS OVERSIGHT", color=YELLOW)
        guard = report.guardian
        guard_icon = "️ " if guard.active else "️ "
        print(f"{guard_icon}  GUARDIAN STATUS:")
        print(f"   - Active: {guard.active}")
        print(f"   - Safe Mode: {' ENABLED' if guard.safe_mode else '🟢 DISABLED'}")
        if guard.paused_features:
            print(f"   - Paused Features: {', '.join(guard.paused_features)}")

        # 3. Remediation Registry
        print_section("REMEDIATION REGISTRY", color=BLUE)
        if not report.remediations:
            print("   (No active remediators registered)")
        else:
            for rem in report.remediations:
                state_color = (
                    GREEN if rem.status == "idle" else (YELLOW if rem.status == "cooldown" else RED)
                )
                print(
                    f"   ▶ {BOLD}{rem.name:<25}{RESET} : {state_color}{rem.status.upper():<10}{RESET} (Last Run: {rem.last_run})"
                )

        print(f"\n{MAGENTA}{'=' * 60}{RESET}")
        print(f"{MAGENTA}{BOLD}  END OF MANIFOLD REPORT{RESET}")
        print(f"{MAGENTA}{'=' * 60}{RESET}")

    except Exception as e:
        import traceback
        if "No module named 'ray'" in str(e):
             print(f"\n{YELLOW}️  RAY NOT AVAILABLE (Incompatible Python version or not installed){RESET}")
        else:
             print(f"\n{RED} ERROR FETCHING MANIFOLD STATE: {str(e)}{RESET}")
             traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_dashboard())
