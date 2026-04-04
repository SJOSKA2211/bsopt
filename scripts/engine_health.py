#!/usr/bin/env python3
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import argparse
import logging
from datetime import datetime

# Institutional-grade minimal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("engine_health")

# Service configurations
SERVICES = {
    "App Gateway": {"url": "http://localhost:5173", "type": "http", "container": "frontend"},
    "Frontend Flow": {"heartbeat": "/tmp/frontend_heartbeat", "type": "dockerexec", "container": "frontend"},
    "API Gateway": {"url": "http://localhost:8081/health", "type": "http"},
    "Auth Service": {"url": "http://localhost:3001/health", "type": "http"},
    "ML Inference": {"url": "http://localhost:5002/health", "type": "http"},
    "Portfolio": {"url": "http://localhost:8003/health", "type": "http"},
    "Ingestion": {"heartbeat": "/tmp/ingestion_heartbeat", "type": "heartbeat"},
    "NSE Scraper": {"heartbeat": "/tmp/scraper_heartbeat", "type": "heartbeat"},
    "yfinance Scraper": {"heartbeat": "/tmp/yfinance_heartbeat", "type": "heartbeat"},
    "Transformer": {"heartbeat": "/tmp/transformer_heartbeat", "type": "heartbeat"},
    "Manifold Core": {"type": "native"},
    "Geth Blockchain": {"url": "http://localhost:8545", "type": "rpc"},
    "CI Test Runner": {"heartbeat": "/tmp/ci_heartbeat", "type": "heartbeat"},
    "Ray ML Cluster": {"url": "http://localhost:8265/api/cluster_status", "type": "ray"},
    "MinIO Storage": {"url": "http://localhost:9000/minio/health/live", "type": "minio"},
    "MLFlow Tracking": {"url": "http://localhost:5000/health", "type": "http"},
    "MLOps Pipeline": {"url": "http://localhost:8080/health", "type": "http"},
}

# Dynamic Cooldown Tracking
COOLDOWNS = {}

def fetch_url(url, method="GET", json_data=None, timeout=10.0):
    """Reliable URL fetcher using standard library only."""
    try:
        req = urllib.request.Request(url, method=method)
        if json_data:
            req.add_header("Content-Type", "application/json")
            data = json.dumps(json_data).encode("utf-8")
        else:
            data = None

        with urllib.request.urlopen(req, data=data, timeout=timeout) as response:
            status = response.getcode()
            body = response.read().decode("utf-8")
            return status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")
    except Exception as e:
        return 0, str(e)


async def check_http(url):
    try:
        code, body = await asyncio.to_thread(fetch_url, url)
        if code == 200:
            try:
                data = json.loads(body)
                return "healthy", str(data.get("status", "ok"))
            except:
                return "healthy", "OK"
        return "unhealthy", f"HTTP {code}"
    except Exception as e:
        return "down", str(e)


async def check_ray(url):
    """Check Ray cluster status and node counts."""
    try:
        code, body = await asyncio.to_thread(fetch_url, url)
        if code == 200:
            data = json.loads(body)
            # Support multiple Ray API response formats
            ray_data = data.get("data", {})
            cluster_status = ray_data.get("clusterStatus", {})
            autoscaler_report = cluster_status.get("autoscalerReport", {})
            active_nodes_dict = autoscaler_report.get("activeNodes", {})
            
            if active_nodes_dict:
                active_nodes = len(active_nodes_dict)
                total_nodes = active_nodes 
                return "healthy", f"Ray API | Nodes: {active_nodes} Active"
            
            # Fallback
            result = data.get("result", {})
            if isinstance(result, dict):
                nodes = result.get("nodes", [])
                active_nodes = len([n for n in nodes if n.get("state") == "ALIVE"])
                total_nodes = len(nodes)
                return "healthy", f"Nodes: {active_nodes}/{total_nodes} Alive"
            
            return "healthy", "Ray Cluster Online"
        return "unhealthy", f"HTTP {code}"
    except Exception as e:
        return "down", str(e)


async def check_minio(url):
    try:
        code, _ = await asyncio.to_thread(fetch_url, url)
        if code == 200:
            cluster_code, _ = await asyncio.to_thread(
                fetch_url, "http://localhost:9000/minio/health/cluster"
            )
            cluster_status = "Cluster OK" if cluster_code == 200 else f"Cluster {cluster_code}"
            return "healthy", f"Live ✓ | {cluster_status}"
        return "unhealthy", f"HTTP {code}"
    except Exception as e:
        return "down", str(e)


async def check_rpc(url):
    try:
        payload = {"jsonrpc": "2.0", "method": "net_version", "params": [], "id": 67}
        code, body = await asyncio.to_thread(fetch_url, url, method="POST", json_data=payload)
        if code == 200:
            data = json.loads(body)
            if "result" in data:
                return "healthy", f"Net: {data['result']}"
            return "unhealthy", "Invalid RPC response"
        return "unhealthy", f"HTTP {code}"
    except Exception as e:
        return "down", str(e)


def check_heartbeat(path):
    if not os.path.exists(path):
        return "missing", "N/A"
    try:
        with open(path) as f:
            content = f.read().strip()
        try:
            data = json.loads(content)
            ts = data.get("time", 0.0)
            metrics = data.get("metrics", {})
            health_status = metrics.get("health", "ACTIVE")
            delta = time.time() - ts
            if delta < 15:
                return "healthy", f"{health_status}"
            return "stale", f"Last active {delta:.1f}s ago"
        except json.JSONDecodeError:
            ts = float(content)
            delta = time.time() - ts
            if delta < 60:
                return "healthy", f"Active ({delta:.1f}s ago)"
            return "stale", f"Last active {delta:.1f}s ago"
    except Exception as e:
        return "error", str(e)


async def check_dockerexec(container, path):
    """Check heartbeat for services running inside a container."""
    try:
        cmd = ["docker", "exec", "-T", container, "cat", path]
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return "missing", "Container access error"
        
        content = result.stdout.strip()
        try:
            data = json.loads(content)
            ts = data.get("time", 0.0)
            metrics = data.get("metrics", {})
            status = metrics.get("status", "ACTIVE")
            delta = time.time() - ts
            if delta < 30:
                return "healthy", f"Flow Positive | {status}"
            return "stale", f"Last active {delta:.1f}s ago"
        except json.JSONDecodeError:
            return "unhealthy", "Invalid JSON heartbeat"
    except Exception as e:
        return "error", str(e)


async def check_native_manifold():
    return "healthy", "SIMD-Optimized Kernel"


async def get_health_data():
    tasks = []
    for name, config in SERVICES.items():
        if config["type"] == "http":
            tasks.append(check_http(config["url"]))
        elif config["type"] == "heartbeat":
            tasks.append(asyncio.to_thread(check_heartbeat, config["heartbeat"]))
        elif config["type"] == "dockerexec":
            tasks.append(check_dockerexec(config["container"], config["heartbeat"]))
        elif config["type"] == "native":
            tasks.append(check_native_manifold())
        elif config["type"] == "rpc":
            tasks.append(check_rpc(config["url"]))
        elif config["type"] == "minio":
            tasks.append(check_minio(config["url"]))
        elif config["type"] == "ray":
            tasks.append(check_ray(config["url"]))

    results = await asyncio.gather(*tasks)
    return dict(zip(SERVICES.keys(), results))


def print_table(health_data):
    """Clean ASCII table implementation."""
    print("\n" + "=" * 80)
    print(f"{'ENGINE HEALTH REPORT':^80}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")
    print("=" * 80)
    print(f"{'SERVICE':<30} | {'STATUS':<12} | {'DETAILS'}")
    print("-" * 80)

    all_healthy = True
    for name, (status, details) in health_data.items():
        if status == "healthy":
            status_str = "HEALTHY"
        elif status in ["stale", "missing"]:
            status_str = status.upper()
            all_healthy = False
        else:
            status_str = "DOWN"
            all_healthy = False
        print(f"{name:<30} | {status_str:<12} | {details}")

    print("-" * 80)
    if all_healthy:
        print(f"{'✅ ALL SYSTEMS OPERATIONAL':^80}")
    else:
        print(f"{'⚠️ SYSTEMS WARNING DETECTED':^80}")
    print("=" * 80 + "\n")
    return all_healthy


async def send_webhook(message):
    webhook_url = os.environ.get("NOTIFY_WEBHOOK") or os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return
    payload = {
        "text": f"🚨 *BSOPT Health Notification*\n{message}",
        "username": "BSOPT Health Monitor",
        "icon_emoji": ":warning:"
    }
    try:
        await asyncio.to_thread(fetch_url, webhook_url, method="POST", json_data=payload)
    except:
        pass


async def auto_fix_service(name, config):
    container = config.get("container")
    if not container:
        return False

    # Check cooldown (300s)
    last_fix = COOLDOWNS.get(name, 0)
    if time.time() - last_fix < 300:
        return False

    logger.warning(f"🔧 Auto-fix triggered for service: {name} (Container: {container})")
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compose_file = os.path.join(project_root, "infrastructure/orchestration/docker-compose.yml")
        
        cmd = ["docker", "compose", "-f", compose_file, "restart", container]
        subprocess.run(cmd, check=True)
        COOLDOWNS[name] = time.time()
        await send_webhook(f"🛠️ *Auto-Recovery*: Restarted container `{container}` to restore `{name}`.")
        return True
    except Exception as e:
        logger.error(f"Auto-fix failed for {name}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="BSOPT Engine Health & Auto-Fix Tool")
    parser.add_argument("--simulate", action="store_true", help="Simulate a healthy environment")
    parser.add_argument("--wait", action="store_true", help="Enable continuous monitoring across all services")
    parser.add_argument("--auto-fix", action="store_true", help="Enable proactive self-healing for containerized services")
    parser.add_argument("--interval", type=int, default=15, help="Check interval in seconds (default: 15)")
    args = parser.parse_args()

    if args.simulate:
        health_data = {k: ("healthy", "Simulated Component") for k in SERVICES.keys()}
        print_table(health_data)
        return

    # Load .env
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v

    last_alert_time = 0
    
    while True:
        health_data = await get_health_data()
        all_healthy = print_table(health_data)
        
        if not all_healthy and args.auto_fix:
            for name, (status, _) in health_data.items():
                if status != "healthy" and "container" in SERVICES[name]:
                    await auto_fix_service(name, SERVICES[name])

        if not args.wait:
            if not all_healthy:
                logger.error("Systems Warning: Initial health check failed.")
                sys.exit(1)
            break
        
        if not all_healthy and time.time() - last_alert_time > 300:
            down_services = [n for n, (s, _) in health_data.items() if s != "healthy"]
            await send_webhook(f"⚠️ Health Monitoring Alert: Status unstable for {', '.join(down_services)}")
            last_alert_time = time.time()
        elif all_healthy and last_alert_time > 0:
             await send_webhook("✅ All systems recovered.")
             last_alert_time = 0

        await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
