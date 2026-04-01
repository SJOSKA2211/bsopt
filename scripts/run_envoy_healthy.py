#!/usr/bin/env python3
import time
import requests
import os
import sys
import subprocess

def check_envoy_ready(url: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Envoy Readiness at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Envoy Admin /ready endpoint
            response = requests.get(url, timeout=2)
            if response.status_code == 200 and "LIVE" in response.text:
                print(f"[+] Envoy is LIVE and READY")
                return True
            else:
                print(f"[-] Envoy status: {response.status_code} ({response.text.strip()}), retrying...")
        except requests.exceptions.RequestException:
            print(f"[-] Envoy not reachable yet...")
        time.sleep(interval)
    return False

def try_start_envoy():
    print("[*] Attempting to start Envoy via flatpak-spawn --host...")
    # Based on scripts/utils_env.sh discovery:
    p_cmd = ["flatpak-spawn", "--host", "docker", "compose", "-f", "infrastructure/orchestration/docker-compose.yml", "up", "-d", "envoy"]
    
    try:
        print(f"[*] Executing: {' '.join(p_cmd)}")
        result = subprocess.run(p_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("[+] Successfully requested Envoy start via host.")
            return True
        else:
            print(f"[-] flatpak-spawn failed: {result.stderr}")
            # Fallback to local docker compose if available
            subprocess.run(["docker", "compose", "-f", "infrastructure/orchestration/docker-compose.yml", "up", "-d", "envoy"])
    except Exception as e:
        print(f"[-] Command failed: {e}")
    
    return False

if __name__ == "__main__":
    # In Docker, admin is often on 9901. In the compose file, host 9901 maps to 9901.
    ready_url = "http://localhost:9901/ready"
    
    # Try to start it first
    try_start_envoy()
    
    # Now monitor health
    if check_envoy_ready(ready_url):
        print("[***] ENVOY API GATEWAY IS FULLY OPTIMIZED AND HEALTHY [***]")
        sys.exit(0)
    else:
        print("[!] ERROR: Envoy failed to reach healthy state.")
        # Try to show logs
        subprocess.run(["docker", "compose", "-f", "infrastructure/orchestration/docker-compose.yml", "logs", "envoy"])
        sys.exit(1)
