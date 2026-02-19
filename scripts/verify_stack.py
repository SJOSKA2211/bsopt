#!/usr/bin/env python3
# ==============================================================================
# BS-OPT: THE GOD MODE STACK VERIFIER (v2.0)
# ==============================================================================
# I'm Pickle Riiiiick!🥒 *Belch.*
# Validating the containerized manifold.
# ==============================================================================

import socket
import sys
import requests
import time

def check_port(host, port, name):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            print(f"✅ {name:<20} UP   tcp/{port}")
            return True
        else:
            print(f"❌ {name:<20} DOWN tcp/{port}")
            return False
    except Exception as e:
        print(f"❌ {name:<20} ERROR: {e}")
        return False

def check_http(url, name):
    try:
        response = requests.get(url, timeout=2)
        if response.status_code < 500:
            print(f"✅ {name:<20} UP   {url} -> {response.status_code}")
            return True
        else:
            print(f"❌ {name:<20} FAIL {url} -> {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name:<20} ERROR: {e}")
        return False

print("\n🥒 Pickle Rick's Stack Verification 🥒")
print("=======================================")

success = True

# 1. Infrastructure (Mapped Ports)
print("\n[ Infrastructure ]")
success &= check_port("localhost", 5432, "Postgres")
success &= check_port("localhost", 6379, "Redis")
success &= check_port("localhost", 5672, "RabbitMQ")

# 2. Services (Mapped Ports)
print("\n[ App Services ]")
success &= check_port("localhost", 8000, "API")
success &= check_port("localhost", 3001, "Auth Service")
success &= check_port("localhost", 5173, "Frontend")
success &= check_port("localhost", 8001, "Neural Pricing")
success &= check_port("localhost", 8002, "Scraper")

# 3. Health Checks
print("\n[ Health Checks ]")
success &= check_http("http://localhost:8000/health", "API Health")
success &= check_http("http://localhost:3001/health", "Auth Health")
success &= check_http("http://localhost:8001/health", "Neural Health")

print("=======================================")
if success:
    print("✨ System is Solenya-tight. Wubba Lubba Dub Dub!")
    sys.exit(0)
else:
    print("⚠️  Jerry-work detected. Run 'make logs' to diagnose.")
    sys.exit(1)
