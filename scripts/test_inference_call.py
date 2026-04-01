#!/usr/bin/env python3
import requests
import json
import uuid
import sys

def test_inference_call():
    url = "http://localhost:5002/predict"
    print(f"[*] Sending Test Inference Request to {url}...")
    
    # Generic Option Data for Black-Scholes
    payload = {
        "requestId": str(uuid.uuid4()),
        "underlying": "BTC",
        "strike": 100000.0,
        "expiry": 0.5, # 180 days
        "interest_rate": 0.05,
        "volatility": 0.5,
        "option_type": "call"
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            print("[+] INFERENCE SUCCESSFUL:")
            print(json.dumps(data, indent=4))
            return True
        else:
            print(f"[!] INFERENCE FAILED: HTTP {resp.status_code}")
            print(resp.text)
            return False
    except Exception as e:
        print(f"[!] ERROR: {e}")
        return False

if __name__ == "__main__":
    if test_inference_call():
        sys.exit(0)
    else:
        sys.exit(1)
