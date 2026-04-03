#!/usr/bin/env python3
import asyncio
import sys

from src.workers.health import get_worker_health


async def check_worker_healthy(timeout: int = 60, interval: int = 5):
    print("[*] Monitoring Worker & Distributed Layer health...")
    start_time = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start_time < timeout:
        try:
            report = await get_worker_health()
            if report["status"] == "healthy":
                print("[+] WORKER CLUSTER IS HEALTHY")
                print(f"    - Broker: {report['celery_broker']}")
                print(f"    - Workers Alive: {report['workers_alive']}")
                print(f"    - Ray Cluster: {report['ray_cluster']}")
                return True
            else:
                print(
                    f"[-] Worker Cluster state: {report['status']} (Ping: {report['workers_alive']}), retrying..."
                )
        except Exception as e:
            print(f"[-] Health check execution failed: {e}")

        await asyncio.sleep(interval)

    print("[!] ERROR: Worker health check timed out after 60 seconds.")
    return False


if __name__ == "__main__":
    if asyncio.run(check_worker_healthy()):
        sys.exit(0)
    else:
        sys.exit(1)
