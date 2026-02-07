from fastapi import Request  # Import Request
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api.main import app
from src.database import get_db
from src.database.models import User
from src.security.auth import get_api_key


def test_api_key_authentication(api_client: TestClient, mock_db_session: Session, mock_redis_and_celery):
    """Test that API Key authentication correctly identifies the user and applies rate limits."""
    # 1. Create a dummy user
    user = User(
        id="test-user-id",
        email="apikey@example.com", 
        tier="pro", 
        is_active=True, 
        is_verified=True
    )
    
    # 2. Use dependency overrides - THE proper way for FastAPI
    async def override_get_api_key(request: Request): # Accept request object
        # Set state so rate limit works
        request.state.user = user
        return user

    # Override get_api_key with our mock function
    app.dependency_overrides[get_api_key] = override_get_api_key
    # Also override get_db to ensure mock session is used
    app.dependency_overrides[get_db] = lambda: mock_db_session
    
    try:
        payload = {
            "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0, 
            "rate": 0.05, "volatility": 0.2, "option_type": "call",
            "dividend_yield": 0.0
        }
        
        response = api_client.post(
            "/api/v1/pricing/price",
            json=payload,
            headers={"X-API-Key": "any-key"} # This header is just to trigger the get_api_key dependency
        )
        
        # 3. Verify success and rate limit headers (Pro tier = 1000)
        assert response.status_code == 200
        assert response.headers.get("X-RateLimit-Limit") == "1000"
        assert response.json()["success"] is True
        
    finally:
        # Clean up overrides
        app.dependency_overrides.clear()
