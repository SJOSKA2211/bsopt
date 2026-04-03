#!/usr/bin/env python3
import os
import sys
import time

import requests


def check_api_healthy(url: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring API health at {url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"[+] API is HEALTHY (Status: {response.status_code})")
                return True
            else:
                print(f"[-] API returned status {response.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"[-] Connection failed: {e}")

        time.sleep(interval)

    print("[!] ERROR: API health check timed out after 60 seconds.")
    return False


if __name__ == "__main__":
    # Get API host from env or default to localhost
    api_host = os.environ.get("API_HOST", "localhost")
    api_port = os.environ.get("API_PORT", "8000")
    api_url = f"http://{api_host}:{api_port}/health"

    if check_api_healthy(api_url):
        sys.exit(0)
    else:
        sys.exit(1)
