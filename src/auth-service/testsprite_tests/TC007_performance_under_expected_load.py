import time

import requests


def percentile(data, percent):
    size = len(data)
    if size == 0:
        return None
    sorted_data = sorted(data)
    k = int(round((percent / 100) * (size - 1)))
    return sorted_data[k]

def test_performance_under_expected_load():
    base_url = "http://localhost:4000"
    auth_endpoint = f"{base_url}/api/auth/login"  # Using a typical auth sub-route 'login' for test
    headers = {
        "Content-Type": "application/json"
    }
    payload = {}  # Changed payload to empty JSON to match generic handler expectations

    latencies = []
    num_requests = 100  # Typical load for performance test

    for _ in range(num_requests):
        start = time.perf_counter()
        try:
            response = requests.post(auth_endpoint, json=payload, headers=headers, timeout=30)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            # Accept 2xx as success
            assert 200 <= response.status_code < 300
        except requests.RequestException as e:
            raise AssertionError(f"Request failed: {e}")

    p95_latency = percentile(latencies, 95)

    assert p95_latency < 200, f"95th percentile latency {p95_latency:.2f}ms exceeds 200ms threshold"

test_performance_under_expected_load()
