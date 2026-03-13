
import asyncio
import sys

import httpx

API_URL = "http://192.168.23.33:8008"
API_V1_URL = f"{API_URL}/api/v1"

async def check_health():
    print("Checking API health...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/health")
            if response.status_code == 200:
                print("✅ API is healthy!")
                return True
            else:
                print(f"❌ API health check failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False

async def test_pricing():
    print("Testing single option pricing...")
    payload = {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "rate": 0.05,
        "volatility": 0.2,
        "option_type": "call",
        "model": "black_scholes"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_V1_URL}/pricing/price", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Pricing successful: {data['price']}")
                return True
            else:
                print(f"❌ Pricing failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Pricing request failed: {e}")
            return False

async def test_batch_pricing():
    print("Testing batch option pricing...")
    payload = {
        "options": [
            {
                "spot": 100.0,
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "rate": 0.05,
                "volatility": 0.2,
                "option_type": "call",
                "model": "black_scholes"
            },
            {
                "spot": 100.0,
                "strike": 110.0,
                "time_to_expiry": 1.0,
                "rate": 0.05,
                "volatility": 0.2,
                "option_type": "call",
                "model": "black_scholes"
            }
        ]
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_V1_URL}/pricing/batch", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Batch pricing successful: {len(data['results'])} results")
                return True
            else:
                print(f"❌ Batch pricing failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Batch pricing request failed: {e}")
            return False

async def main():
    print("--- BSOPT Verification Script ---")
    
    # Wait for API to be ready
    max_retries = 30
    ready = False
    for i in range(max_retries):
        if await check_health():
            ready = True
            break
        print(f"Waiting for API... ({i+1}/{max_retries})")
        await asyncio.sleep(5)
    
    if not ready:
        print("❌ API did not become ready in time.")
        sys.exit(1)
        
    results = await asyncio.gather(
        test_pricing(),
        test_batch_pricing()
    )
    
    if all(results):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
