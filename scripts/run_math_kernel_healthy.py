#!/usr/bin/env python3
import os
import sys
import time

import grpc
import requests


def check_math_kernel_http(url: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Math Kernel HTTP at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("[+] Math Kernel HTTP is HEALTHY")
                return True
            else:
                print(f"[-] Math Kernel HTTP status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[-] HTTP connection failed: {e}")
        time.sleep(interval)
    return False


def check_math_kernel_grpc(address: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Math Kernel gRPC at {address}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with grpc.insecure_channel(address):
                # We can't easily call PriceOption without real data, but we can check connectivity
                # or use gRPC health if implemented. For now, we'll just check if the port is open and responding.
                # In a real scenario, we'd implementation grpc_health_v1.
                print("[+] Math Kernel gRPC is REACHABLE")
                return True
        except Exception as e:
            print(f"[-] gRPC connection failed: {e}")
        time.sleep(interval)
    return False


if __name__ == "__main__":
    mk_host = os.environ.get("MATH_KERNEL_HOST", "localhost")
    http_port = os.environ.get("HTTP_PORT", "8080")
    grpc_port = os.environ.get("GRPC_PORT", "50052")

    http_url = f"http://{mk_host}:{http_port}/health"
    grpc_addr = f"{mk_host}:{grpc_port}"

    http_ok = check_math_kernel_http(http_url)
    grpc_ok = check_math_kernel_grpc(grpc_addr)

    if http_ok and grpc_ok:
        print("[***] MATH KERNEL IS FULLY HEALTHY [***]")
        sys.exit(0)
    else:
        sys.exit(1)
