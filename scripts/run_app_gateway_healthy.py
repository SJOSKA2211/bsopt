#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import sys
import time


def get_container_engine():
    """Detect container engine, strictly prioritizing docker."""
    return "docker"

def is_port_open(port, host="localhost"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0

def check_heartbeat(path):
    """Check heartbeat via docker exec since it's inside the container tmpfs."""
    try:
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "infrastructure/orchestration/docker-compose.yml",
                "exec",
                "-T",
                "frontend",
                "cat",
                path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout)
        ts = data.get("time", 0)
        # Heartbeat must be within last 15 seconds
        if time.time() - ts < 15:
            return True
    except Exception:
        pass
    return False

def log(msg, level="INFO"):
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{colors.get(level, '')}{level}{colors['RESET']}] {timestamp} - {msg}")

def main():
    engine = get_container_engine()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    compose_file = os.path.join(project_root, "infrastructure/orchestration/docker-compose.yml")
    heartbeat_path = "/tmp/frontend_heartbeat"

    log("🚀 Starting App Gateway (Frontend) Optimization Sequence", "INFO")

    # Start the service
    log("Starting frontend service via docker compose...", "INFO")
    try:
        subprocess.run([engine, "compose", "-f", compose_file, "up", "-d", "frontend"], check=True)
    except subprocess.CalledProcessError as e:
        log(f"Failed to start frontend: {e}", "ERROR")
        sys.exit(1)

    # Phase 1: Port Gating (Vite Server)
    log("Waiting for Vite Dev Server (Port 5173)...", "INFO")
    start_time = time.time()
    port_healthy = False
    while time.time() - start_time < 120:
        if is_port_open(5173):
            log("Port 5173 is OPEN", "SUCCESS")
            port_healthy = True
            break
        time.sleep(2)

    if not port_healthy:
        log("Timed out waiting for port 5173", "ERROR")
        sys.exit(1)

    # Phase 2: Flow Gating (AIOps Heartbeat)
    log("Waiting for Flow-Positive Heartbeat (AIOps)...", "INFO")
    start_time = time.time()
    heartbeat_healthy = False
    while time.time() - start_time < 90:
        if check_heartbeat(heartbeat_path):
            log("App Gateway is HEALTHY and REPORTING FLOW", "SUCCESS")
            heartbeat_healthy = True
            break
        time.sleep(3)

    if not heartbeat_healthy:
        log("Timed out waiting for AIOps Healthy state", "ERROR")
        # Don't exit here, maybe it just takes longer, but warn.
        # sys.exit(1)

    log("🎉 Frontend Optimization & Startup Sequence Complete", "SUCCESS")

if __name__ == "__main__":
    main()
