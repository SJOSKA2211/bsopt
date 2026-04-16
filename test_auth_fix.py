import asyncio
import os
import uuid

# Set environment variables for testing
os.environ["BSOPT_ALLOW_WEAK_SECRETS"] = "true"
os.environ["JWT_SECRET"] = "a" * 32
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

from src.auth.auth import auth_service
from src.auth.core.tokens import token_service
from src.shared.utils.cache import init_redis_cache


async def test_auth_validation():
    # Initialize redis mock/connection
    await init_redis_cache()
    
    user_id = str(uuid.uuid4())
    email = "test@example.com"
    tier = "pro"
    
    # 1. Create token
    token_pair = token_service.create_token_pair(user_id, email, tier)
    token = token_pair.access_token
    print(f"Created token: {token[:20]}...")
    
    # 2. Validate token (This should now work without calling non-existent db_cache.get)
    try:
        token_data = await auth_service.validate_token(token)
        print(f"Validated token for user: {token_data.user_id}")
        assert token_data.user_id == user_id
        assert token_data.email == email
        print("Auth validation test PASSED")
    except Exception as e:
        print(f"Auth validation test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_auth_validation())
