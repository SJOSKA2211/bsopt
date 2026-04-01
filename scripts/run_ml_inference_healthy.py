#!/usr/bin/env python3
import asyncio
import httpx
import sys
import os

async def check_ml_inference_healthy(timeout: int = 60, interval: int = 5):
    url = "http://localhost:5002/health"
    print(f"[*] Monitoring ML Inference Health at {url}...")
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("models", {})
                    if data.get("status") == "healthy" and models.get("xgb"):
                        print("[+] ML INFERENCE IS HEALTHY AND MODELS ARE LOADED")
                        print(f"    - XGB Model: {models.get('xgb')}")
                        print(f"    - NN Model: {models.get('nn')}")
                        return True
                    else:
                        print(f"[-] Service OK but models not ready: {models}")
                else:
                    print(f"[-] Service reported {resp.status_code}, retrying...")
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            
        await asyncio.sleep(interval)
        
    print("[!] ERROR: ML Inference health check timed out.")
    return False

if __name__ == "__main__":
    if asyncio.run(check_ml_inference_healthy()):
        sys.exit(0)
    else:
        sys.exit(1)
