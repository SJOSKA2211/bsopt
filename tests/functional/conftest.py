"""
Fixtures for Functional Tests (Principles 27, 53, 93, 94)
======================================================
"""

import pytest
import pytest_asyncio
import uuid
import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
from faker import Faker
from sqlalchemy.orm import Session

from src.api.main import app
from src.database import get_db
from src.database.models import User

fake = Faker()

# 9. Test Data Management & 28. Data Generation
@pytest.fixture
def user_payload():
    """Generates unique, valid user registration data."""
    uid = str(uuid.uuid4())[:8]
    password = f"Complexity_Is_Key_{uid}2025!"
    return {
        "email": f"func_{uid}@example.com",
        "password": password,
        "password_confirm": password,
        "full_name": fake.name(),
        "accept_terms": True,
    }

# 10. Database State & 42. Test Isolation
@pytest.fixture
def mock_db():
    """
    Mock DB session with in-memory user store.
    Ensures a clean state per test (41. Cleanup).
    """
    session = MagicMock(spec=Session)
    users = {}

    def mock_add(obj):
        if isinstance(obj, User):
            if obj.email in users:
                from sqlalchemy.exc import IntegrityError
                raise IntegrityError("duplicate key", params={}, orig=None)
            
            if getattr(obj, "id", None) is None: obj.id = uuid.uuid4()
            if getattr(obj, "created_at", None) is None: obj.created_at = datetime.now(timezone.utc)
            if getattr(obj, "is_mfa_enabled", None) is None: obj.is_mfa_enabled = False
            if getattr(obj, "is_active", None) is None: obj.is_active = True
            if getattr(obj, "tier", None) is None: obj.tier = "free"
            users[str(obj.id)] = obj
            users[obj.email] = obj
    
    def mock_query(model):
        mq = MagicMock()
        mq._filter_val = None
        def filter_se(cond):
            try:
                mq._filter_val = cond.right.value
            except AttributeError:
                try:
                    import re
                    s = str(cond)
                    match = re.search(r"'(.*?)'", s)
                    if match: mq._filter_val = match.group(1)
                    else:
                        uuid_match = re.search(r"[0-9a-f-]{36}", s)
                        if uuid_match: mq._filter_val = uuid_match.group(0)
                except: pass
            return mq
        mq.filter.side_effect = filter_se
        mq.first.side_effect = lambda: users.get(str(mq._filter_val))
        return mq

    session.add.side_effect = mock_add
    session.query.side_effect = mock_query
    session.users = users 
    return session

# 11. API Client (httpx.AsyncClient)
@pytest_asyncio.fixture
async def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    
    # 69. Test Framework: mock.patch audit logs
    with patch("src.security.audit.log_audit"), \
         patch("src.security.audit.AuditLog"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            yield ac
    app.dependency_overrides.clear()