#!/usr/bin/env python3
import os
import socket
import sys
import time


def check_ingestion_healthy(
    heartbeat_file: str, grpc_port: int, timeout: int = 60, interval: int = 5
):
    print("[*] Monitoring Ingestion Service health...")
    print(f"    - Heartbeat: {heartbeat_file}")
    print(f"    - gRPC Port: {grpc_port}")

    start_time = time.time()
    while time.time() - start_time < timeout:
        # 1. Check Heartbeat File
        if os.path.exists(heartbeat_file):
            age = time.time() - os.path.getmtime(heartbeat_file)
            if age < 60:
                print(f"[+] Ingestion HEARTBEAT is FRESH (Age: {age:.2f}s)")
                return True
            else:
                print(f"[-] Ingestion HEARTBEAT is STALE (Age: {age:.2f}s), retrying...")
        else:
            # 2. Fallback: Check gRPC Port
            try:
                with socket.create_connection(("localhost", grpc_port), timeout=2.0):
                    print(f"[+] Ingestion gRPC Port {grpc_port} is OPEN (Fallback)")
                    return True
            except Exception:
                print(
                    f"[-] Ingestion Heartbeat missing and gRPC Port {grpc_port} closed, retrying..."
                )

        time.sleep(interval)

    print("[!] ERROR: Ingestion health check timed out after 60 seconds.")
    return False


if __name__ == "__main__":
    heartbeat_path = os.environ.get("INGESTION_HEARTBEAT", "/tmp/ingestion_heartbeat")
    grpc_port = int(os.environ.get("INGESTION_PORT", 50053))

    if check_ingestion_healthy(heartbeat_path, grpc_port):
        sys.exit(0)
    else:
        sys.exit(1)
