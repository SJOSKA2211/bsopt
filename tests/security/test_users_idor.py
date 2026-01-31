
import sys
from unittest.mock import MagicMock, AsyncMock

# Mock numba before importing anything else
sys.modules["numba"] = MagicMock()

import pytest
from fastapi.testclient import TestClient

# Mock redis initialization to avoid connection errors during app startup
with pytest.MonkeyPatch.context() as mp:
    mock_redis = AsyncMock()
    # Mock init_redis_cache to return a mock redis client
    # We need to patch where it is imported in main.py or the module itself
    # Since main.py imports it inside lifespan, we patch the module function
    import src.utils.cache
    mp.setattr(src.utils.cache, "init_redis_cache", AsyncMock(return_value=mock_redis))
    mp.setattr(src.utils.cache, "warm_cache", AsyncMock())

    # Also patch preload functions to speed up test
    import src.ml
    mp.setattr(src.ml, "preload_critical_modules", MagicMock())
    import src.pricing
    mp.setattr(src.pricing, "preload_classical_pricers", MagicMock())

    from src.api.main import app
    from src.database.models import User
    from src.database import get_async_db
    from src.security.auth import get_current_active_user
    import uuid
    from datetime import datetime, timezone

    client = TestClient(app)

    @pytest.fixture
    def enterprise_user():
        return User(
            id=uuid.uuid4(),
            email="enterprise@example.com",
            full_name="Enterprise User",
            tier="enterprise", # This user has enterprise tier
            is_active=True,
            is_verified=True,
            is_mfa_enabled=False,
            created_at=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def victim_user():
        return User(
            id=uuid.uuid4(),
            email="victim@example.com",
            full_name="Victim User",
            tier="free",
            is_active=True,
            is_verified=True,
            is_mfa_enabled=False,
            created_at=datetime.now(timezone.utc)
        )

    @pytest.mark.asyncio
    async def test_idor_vulnerability(enterprise_user, victim_user):
        """
        Test that an enterprise user can access another user's profile (IDOR).
        If this test passes, the vulnerability exists.
        """
        # Mock authentication to return the enterprise user
        app.dependency_overrides[get_current_active_user] = lambda: enterprise_user

        # Mock the database to return the victim user when queried by ID
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = victim_user
        mock_db.execute.return_value = mock_result

        app.dependency_overrides[get_async_db] = lambda: mock_db

        # The enterprise user tries to access the victim user's profile
        response = client.get(f"/api/v1/users/{victim_user.id}")

        # Clean up overrides
        app.dependency_overrides = {}

        # With the fix, the status code should be 403 Forbidden.
        assert response.status_code == 403
        assert "permission" in response.json()["message"].lower()
