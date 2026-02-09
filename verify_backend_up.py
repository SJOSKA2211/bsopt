import time
import requests
import sys

URL = "http://localhost:8000/health"
MAX_RETRIES = 30
SLEEP_TIME = 2

print(f"Checking {URL}...")
for i in range(MAX_RETRIES):
    try:
        response = requests.get(URL)
        if response.status_code == 200:
            print("Backend is up!")
            sys.exit(0)
        else:
            print(f"Backend returned status code {response.status_code}")
    except requests.ConnectionError:
        print(f"Connection failed (attempt {i+1}/{MAX_RETRIES})")
    time.sleep(SLEEP_TIME)

print("Backend failed to start within timeout")
sys.exit(1)
