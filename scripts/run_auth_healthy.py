#!/usr/bin/env python3
import time
import requests
import os
import sys
import socket
import grpc
import json
from grpc_health.v1 import health_pb2, health_pb2_grpc

def check_http_readiness(url: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Auth HTTP Readiness at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                health_report = response.json()
                print(f"[+] Auth HTTP is READY (Status: {response.status_code})")
                print(f"    - Health Report: {json.dumps(health_report, indent=2)}")
                return True
            else:
                try:
                    error_data = response.json()
                    print(f"[-] Auth HTTP returned status {response.status_code} (Not Ready): {error_data}")
                except:
                    print(f"[-] Auth HTTP returned status {response.status_code} (Not Ready), retrying...")
        except requests.exceptions.RequestException as e:
            print(f"[-] Auth HTTP connection failed: {e}")
        time.sleep(interval)
    return False

def check_grpc_health(address: str, timeout: int = 60, interval: int = 5):
    print(f"[*] Monitoring Auth gRPC Health at {address}...")
    start_time = time.time()
    
    # Try Secure Channel first (TLS)
    ca_cert_path = os.path.join(os.getcwd(), ".pki/root_ca.crt")
    client_cert_path = os.path.join(os.getcwd(), ".pki/auth-service.crt")
    client_key_path = os.path.join(os.getcwd(), ".pki/auth-service.key")
    
    creds = None
    if os.path.exists(ca_cert_path):
        with open(ca_cert_path, 'rb') as f:
            ca_cert = f.read()
        
        client_cert = None
        client_key = None
        if os.path.exists(client_cert_path) and os.path.exists(client_key_path):
            with open(client_cert_path, 'rb') as f:
                client_cert = f.read()
            with open(client_key_path, 'rb') as f:
                client_key = f.read()
        
        if client_cert and client_key:
            creds = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=client_key,
                certificate_chain=client_cert
            )
            print(f"[*] Using mTLS for gRPC health check (Cert: {client_cert_path})")
        else:
            creds = grpc.ssl_channel_credentials(root_certificates=ca_cert)
            print(f"[*] Using TLS for gRPC health check (CA: {ca_cert_path})")

    while time.time() - start_time < timeout:
        try:
            if creds:
                # Override target name for hostname verification since we're using 'auth-service' cert on localhost
                options = (('grpc.ssl_target_name_override', 'auth-service'),)
                channel = grpc.secure_channel(address, creds, options=options)
            else:
                channel = grpc.insecure_channel(address)
            
            with channel:
                stub = health_pb2_grpc.HealthStub(channel)
                # Note: We use an empty service name or 'auth.AuthService'
                request = health_pb2.HealthCheckRequest(service="auth.AuthService")
                response = stub.Check(request, timeout=2.0)
                if response.status == health_pb2.HealthCheckResponse.SERVING:
                    print(f"[+] Auth gRPC is SERVING (Secure: {bool(creds)})")
                    return True
                else:
                    print(f"[-] Auth gRPC status: {response.status}")
        except grpc.RpcError as e:
            # If secure fails and we haven't tried insecure yet, or just report error
            print(f"[-] Auth gRPC connection failed: {e.code()} - {e.details()}")
            
            # Fallback to insecure if secure failed with certain codes (optional)
            if creds and e.code() == grpc.StatusCode.UNAVAILABLE:
                pass # Continue loop
        except Exception as e:
            print(f"[-] Auth gRPC unexpected error: {e}")
            
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
