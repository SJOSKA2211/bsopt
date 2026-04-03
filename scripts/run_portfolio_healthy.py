#!/usr/bin/env python3
import os
import sys
import time

import requests


def check_portfolio_healthy(url: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Portfolio & Risk health at {url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"[+] Portfolio Service is HEALTHY (Status: {response.status_code})")
                return True
            else:
                print(f"[-] Portfolio Service returned status {response.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"[-] Connection failed: {e}")

        time.sleep(interval)

    print("[!] ERROR: Portfolio health check timed out after 60 seconds.")
    return False


if __name__ == "__main__":
    # Get Portfolio host from env or default to localhost
    port_host = os.environ.get("PORTFOLIO_HOST", "localhost")
    port_number = os.environ.get("PORTFOLIO_PORT", "8080")
    port_url = f"http://{port_host}:{port_number}/health/readiness"

    if check_portfolio_healthy(port_url):
        sys.exit(0)
    else:
        sys.exit(1)
