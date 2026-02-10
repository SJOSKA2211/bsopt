#!/usr/bin/env python3
import socket
import sys

import requests


def check_port(host, port, name):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            print(f"✅ {name} is UP ({host}:{port})")
            return True
        else:
            print(f"❌ {name} is DOWN ({host}:{port})")
            return False
    except Exception as e:
        print(f"❌ {name} check failed: {e}")
        return False

def check_http(url, name):
    try:
        response = requests.get(url, timeout=2)
        if response.status_code < 500: # 404/401 is technically "up"
            print(f"✅ {name} HTTP is UP ({url} -> {response.status_code})")
            return True
        else:
            print(f"❌ {name} HTTP returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name} HTTP check failed: {e}")
        return False

print(" Joseph Kamau Maina's Stack Verification ")
print("---------------------------------------")

success = True

# 1. Infrastructure
success &= check_port("localhost", 5432, "Postgres")
success &= check_port("localhost", 6379, "Redis")
success &= check_port("localhost", 5672, "RabbitMQ")

# 2. Services (Check if they are running locally - User must have started them!)
print("\nChecking App Services (Must be started manually via scripts/start_*.sh)...")
api_up = check_port("localhost", 8000, "API")
auth_up = check_port("localhost", 3001, "Auth Service")
front_up = check_port("localhost", 5173, "Frontend")
neural_up = check_port("localhost", 8001, "Neural Pricing")

if api_up:
    success &= check_http("http://localhost:8000/health", "API Health")
if auth_up:
    success &= check_http("http://localhost:3001/", "Auth Root")
if neural_up:
    success &= check_http("http://localhost:8001/health", "Neural Pricing Health")

print("---------------------------------------")
if success:
    print("System check complete. The stack is operational. ")
    sys.exit(0)
else:
    print("Jerry-work detected. Something is broken.")
    sys.exit(1)
