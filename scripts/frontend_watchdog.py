#!/usr/bin/env python3
import json
import logging
import os
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# Institutional-grade minimal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("frontend_watchdog")

# Configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
HEARTBEAT_PATH = "/tmp/frontend_heartbeat"
CHECK_INTERVAL = 30
STALE_THRESHOLD = 90  # Seconds before considering a heartbeat stale
HEALTH_PORT = 8081  # Watchdog's own health server
CONTAINER_NAME = "frontend"

# Global health status for the watchdog itself
IS_HEALTHY = False
HEALTH_DETAILS = "Initializing..."

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            if IS_HEALTHY:
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(503)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Frontend Degraded: {HEALTH_DETAILS}".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

def is_port_open(port, host="localhost"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        return s.connect_ex((host, port)) == 0

def get_heartbeat_age():
    """Exec into container to get heartbeat age."""
    try:
        cmd = ["docker", "exec", "-T", CONTAINER_NAME, "cat", HEARTBEAT_PATH]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None, "Container access failed"
        
        data = json.loads(result.stdout)
        ts = data.get("time", 0)
        return time.time() - ts, "OK"
    except Exception as e:
        return None, str(e)

def send_alert(message):
    webhook_url = os.environ.get("NOTIFY_WEBHOOK") or os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return
    
    payload = {"text": f"🚨 *Frontend Watchdog*: {message}"}
    try:
        req = urllib.request.Request(webhook_url, method="POST")
        req.add_header("Content-Type", "application/json")
        data = json.dumps(payload).encode("utf-8")
        with urllib.request.urlopen(req, data=data, timeout=5) as _:
            pass
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

def restart_frontend():
    logger.warning("🔄 Initiating Frontend restart sequence...")
    send_alert("Stale heartbeat or port failure detected. Restarting container...")
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        compose_file = os.path.join(project_root, "infrastructure/orchestration/docker-compose.yml")
        subprocess.run(["docker", "compose", "-f", compose_file, "restart", "frontend"], check=True)
        logger.info("✅ Frontend restart command issued.")
    except Exception as e:
        logger.error(f"Failed to restart frontend: {e}")
        send_alert(f"RESTART FAILED: {e}")

class FrontendWatchdog:
    def __init__(self):
        self.consecutive_failures = 0
        self.max_failures = 3

    def start_health_server(self):
        def serve():
            try:
                server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
                logger.info(f"Frontend Watchdog health server listening on port {HEALTH_PORT}")
                server.serve_forever()
            except Exception as e:
                logger.error(f"Watchdog health server failed: {e}")

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

    def run(self):
        global IS_HEALTHY, HEALTH_DETAILS
        logger.info("🚀 BSOPT Frontend Watchdog starting...")
        self.start_health_server()

        while True:
            port_up = is_port_open(5173)
            age, status = get_heartbeat_age()
            
            logic_up = age is not None and age < STALE_THRESHOLD
            
            if port_up and logic_up:
                if not IS_HEALTHY:
                    logger.info("🎉 Frontend recovered and healthy.")
                    send_alert("Frontend fully recovered.")
                IS_HEALTHY = True
                HEALTH_DETAILS = f"Flow Positive (Age: {age:.1f}s)"
                self.consecutive_failures = 0
            else:
                IS_HEALTHY = False
                reasons = []
                if not port_up:
                    reasons.append("Vite Port 5173 Unreachable")
                if not logic_up:
                    age_str = f"{age:.1f}s" if age else "N/A"
                    reasons.append(f"Heartbeat Stale ({age_str}) | {status}")
                
                HEALTH_DETAILS = " | ".join(reasons)
                logger.error(f"⚠️ Health Violation: {HEALTH_DETAILS}")
                
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    restart_frontend()
                    self.consecutive_failures = 0
                    time.sleep(60) # Wait for startup before next check
            
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    FrontendWatchdog().run()
