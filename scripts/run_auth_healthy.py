#!/usr/bin/env python3
import time
import requests
import os
import sys

def check_auth_healthy(url: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Auth Service health at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"[+] Auth Service is HEALTHY (Status: {response.status_code})")
                return True
            else:
                print(f"[-] Auth Service returned status {response.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"[-] Connection failed: {e}")
            
        time.sleep(interval)
        
    print("[!] ERROR: Auth health check timed out after 60 seconds.")
    return False

if __name__ == "__main__":
    # Get Auth host from env or default to localhost
    auth_host = os.environ.get("AUTH_HOST", "localhost")
    auth_port = os.environ.get("AUTH_PORT", "3001")
    auth_url = f"http://{auth_host}:{auth_port}/health"
    
    if check_auth_healthy(auth_url):
        sys.exit(0)
    else:
        sys.exit(1)
