#!/usr/bin/env python3
import time
import requests
import os
import sys

def check_neural_pricing_healthy(url: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Neural Pricing health at {url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy" and data.get("model_loaded"):
                    print(f"[+] Neural Pricing is HEALTHY and Model is LOADED")
                    return True
                else:
                    print(f"[-] Neural Pricing responded: {data}")
            else:
                print(f"[-] Neural Pricing returned status {response.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"[-] Connection failed: {e}")
            
        time.sleep(interval)
        
    print("[!] ERROR: Neural Pricing health check timed out after 60 seconds.")
    return False

if __name__ == "__main__":
    # Get Neural Pricing host from env or default to localhost
    np_host = os.environ.get("NEURAL_PRICING_HOST", "localhost")
    np_port = os.environ.get("NEURAL_PRICING_PORT", "5001")
    np_url = f"http://{np_host}:{np_port}/health/readiness"
    
    if check_neural_pricing_healthy(np_url):
        sys.exit(0)
    else:
        sys.exit(1)
