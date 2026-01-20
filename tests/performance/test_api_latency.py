import time
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

@pytest.mark.performance
def test_pricing_api_latency():
    """Benchmark the latency of the core pricing API endpoint."""
    payload = {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "rate": 0.05,
        "volatility": 0.2,
        "option_type": "call",
        "model": "black_scholes"
    }
    
    # Warmup
    client.post("/api/v1/pricing/price", json=payload)
    
    start_time = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        response = client.post("/api/v1/pricing/price", json=payload)
        assert response.status_code == 200
        
    avg_latency = (time.perf_counter() - start_time) / iterations * 1000
    print(f"\nAverage API Latency: {avg_latency:.2f}ms")
    assert avg_latency < 50  # Target < 50ms for Black-Scholes analytical
