import uuid
import requests

# Point to the new auth-service (Node.js)
BASE_URL = "http://127.0.0.1:3001"
# better-auth standard endpoints
REGISTER_ENDPOINT = "/api/auth/sign-up/email"
TIMEOUT = 30

def test_post_auth_register_better_auth():
    unique_email = f"testuser_{uuid.uuid4()}@example.com"
    # better-auth sign-up/email expects: email, password, name
    payload = {
        "email": unique_email, 
        "password": "P@ssw0rd123!", 
        "name": "Test User BetterAuth"
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(
            f"{BASE_URL}{REGISTER_ENDPOINT}",
            json=payload,
            headers=headers,
            timeout=TIMEOUT,
        )
        # better-auth returns 200/201 on success depending on config
        assert response.status_code in [200, 201], f"Expected success, got {response.status_code}: {response.text}"
        
        json_response = response.json()
        assert "token" in json_response or "session" in json_response or "user" in json_response
        
        # Verify rate limit headers are present (from our new middleware)
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        
    except Exception as e:
        assert False, f"BetterAuth registration failed: {e}"

if __name__ == "__main__":
    test_post_auth_register_better_auth()
    print("TC011: BetterAuth Registration Test Passed")
