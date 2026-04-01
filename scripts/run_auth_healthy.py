#!/usr/bin/env python3
import time
import requests
import os
import sys
import socket
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

def check_http_readiness(url: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Auth HTTP Readiness at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"[+] Auth HTTP is READY (Status: {response.status_code})")
                return True
            else:
                print(f"[-] Auth HTTP returned status {response.status_code} (Not Ready), retrying...")
        except requests.exceptions.RequestException as e:
            print(f"[-] Auth HTTP connection failed: {e}")
        time.sleep(interval)
    return False

def check_grpc_health(address: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Auth gRPC Health at {address}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with grpc.insecure_channel(address) as channel:
                stub = health_pb2_grpc.HealthStub(channel)
                request = health_pb2.HealthCheckRequest(service="auth.AuthService")
                response = stub.Check(request, timeout=2.0)
                if response.status == health_pb2.HealthCheckResponse.SERVING:
                    print(f"[+] Auth gRPC is SERVING")
                    return True
                else:
                    print(f"[-] Auth gRPC status: {response.status}")
        except grpc.RpcError as e:
            print(f"[-] Auth gRPC connection failed: {e.code()}")
        time.sleep(interval)
    return False

if __name__ == "__main__":
    auth_host = os.environ.get("AUTH_HOST", "localhost")
    http_port = os.environ.get("HTTP_PORT", "3001")
    grpc_port = os.environ.get("GRPC_PORT", "50051")
    
    http_url = f"http://{auth_host}:{http_port}/health/readiness"
    grpc_addr = f"{auth_host}:{grpc_port}"
    
    http_ok = check_http_readiness(http_url)
    grpc_ok = check_grpc_health(grpc_addr)
    
    if http_ok and grpc_ok:
        print("[***] AUTH SERVICE MANIFOLD IS FULLY HEALTHY [***]")
        sys.exit(0)
    else:
        print("[!] ERROR: Auth Service Manifold failed health checks.")
        sys.exit(1)
