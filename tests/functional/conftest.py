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

# PRE-MOCK SETTINGS
from unittest.mock import MagicMock
import sys
mock_settings = MagicMock()
mock_settings.BCRYPT_ROUNDS = 12
mock_settings.JWT_ALGORITHM = "RS256"
mock_settings.JWT_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCxnuu6P25C2gbg
K4Q6QKgFcjHCJ/+M3raqsvz1kiZCpIgP/qKGKu70j4KcwsJrWPxop10x/KhbHpSv
8OCnF3kMWCNxHw4QLJ/u5Aufbe++Irayic9j9lIn1cyWcWk1E9qtCWJQ/zBdgucC
2stZdLMpMWmAzIsVcqTgSU2OosGwf6Uf77dqCLSXFKf2+khn8BiH2mXAK/bLFGyh
4biPlzaTSIcNmdnJPH4iWzC9WJARqjr9ZRGkWKbgn+7lGJeHRxf6axRv8t+p9isH
xuQzALa8iO6HhxiF3H5cduDPghU6DkmKl6V/bSB5qFsP2FHjWfMnPipTvbeuEKCk
nzIEptDJAgMBAAECggEABr5mLE7auJqPDsVQMqcASh6lEX5Tx00wacg8bvVy0u5s
xRCxqn7oTixBtQJ2/7zj7nRGR07UtIrzcb+nQ+jR/Xw+Mj4P2mDbXKZXY6D4rIMk
ZSBy2ZSBV4ZYS3D4Yd3EXHQCAdnChBZjf3n/pQCXic2YuB1r/W86H9LgqTT4PiN2
V/WOf5eiBZsQVj4bsOlMPdO+6hmwnx86raUbKE+FJYAhvqWlesjUeNm4r6Yfotct
Ig7IltZnUvBcYgxKr2tvzzlUtsiiCx4qIXphqO9zVzg9U4B5Wc4X72xHVP0kZG7O
afxFZSkLzwPfSqsVJXXkW6xF0GHXw30H8raO648/9QKBgQDWObYgEtq58GOD5peN
p0yHgwZVpRoko55Aj0LmlWzpE2Ny5yEvV/EZNSZi+aaCN+a6Wqgyfgoofg2hIVZe
nvdUvXtCeLvw2Fe6u/hbEzhIXKOclAQ6cwtisyfYwSfi8j6oMXuhuukYthUdhRAp
uddaUy+jVzDiiN6EPUlHWCqm5QKBgQDUQeD2G1BHbd2eQhWFsDRVaSQiBfe+KaM0
SmfMQAFBsQtnQGIsNifzP1bAZyVHs1shPRJaUIPs0i5e53eOaF0C1EovHW0EFwSn
UvtNMAb0uRZqpOcrGsUzRX1asgrv1fMxOffWdFmK6JGjzd/3wEGedFyQ2O+wTfwG
xeKLd/igFQKBgA1Tp8XVBnBcyQQSm0j/qF4hw4oebELtPtILV4EauJzDTQN/52uX
j/MegFXV7ArbyWm8bAxAFQex1803UrUuNHq8EufutNplywdd3DRmPLEbuj3qY1zz
fTjVplvwoDeZFFbIRUWpaAjWgvfEKF5AJmqDFEqYCP1+wED/wwhCLt0VAoGBAJ/K
LIn5y/jKC9HNHBi1quA1s97tMTF2dQezj+qisI98sgH75Sw1ZOPpZeyYeec9bbhb
GorlHDvXitMlW8rYZFTx7hsEAwLWNUml3cuhAUuQXwDPvbukfpp3kMQLTtJ49Yi0
hBBtLM+2/5UaMqZ3lK6uGNVuixrlynpq1H58Ra51AoGALlRt4xPMOMk+4F9mVk2R
3aZczsnsfi3o6En0hHSZnoftJZHxAGEFPC4p8zavMJrtUM/EE1fKnYR8+JJyYFCA
pbZ7HoWVhxY/QluS6GdDs74A90BDLFTI9SNfBcYf7Gn981i5+ofysSpoII3SEj14
/5M82ANrBoH+QMxDbY2rnLs=
-----END PRIVATE KEY-----"""
mock_settings.JWT_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAsZ7ruj9uQtoG4CuEOkCo
BXIxwif/jN62qrL89ZImQqSID/6ihiru9I+CnMLCa1j8aKddMfyoWx6Ur/Dgpxd5
DFgjcR8OECyf7uQLn23vviK2sonPY/ZSJ9XMlnFpNRParQliUP8wXYLnAtrLWXSz
KTFpgMyLFXKk4ElNjqLBsH+lH++3agi0lxSn9vpIZ/AYh9plwCv2yxRsoeG4j5c2
k0iHDZnZyTx+IlswvViQEao6/WURpFim4J/u5RiXh0cX+msUb/LfqfYrB8bkMwC2
vIjuh4cYhdx+XHbgz4IVOg5Jipelf20geahbD9hR41nzJz4qU723rhCgpJ8yBKbQ
yQIDAQAB
-----END PUBLIC KEY-----"""
mock_settings.DEBUG = True
mock_settings.is_production = False
mock_settings.ENVIRONMENT = "dev"
mock_settings.CORS_ORIGINS = ["http://localhost:3000"]
mock_settings.PASSWORD_MIN_LENGTH = 8
mock_settings.PASSWORD_REQUIRE_UPPERCASE = True
mock_settings.PASSWORD_REQUIRE_LOWERCASE = True
mock_settings.PASSWORD_REQUIRE_DIGIT = True
mock_settings.PASSWORD_REQUIRE_SPECIAL = False
mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30
mock_settings.REFRESH_TOKEN_EXPIRE_DAYS = 7
mock_settings.JWT_SECRET = "test-secret-key-that-is-long-enough-32-chars"
mock_settings.rate_limit_tiers = {"free": 100, "pro": 1000, "enterprise": 10000}

if "src.config" in sys.modules:
    sys.modules["src.config"].settings = mock_settings
    sys.modules["src.config"]._settings = mock_settings
    sys.modules["src.config"].get_settings = lambda: mock_settings
else:
    import types
    cfg = types.ModuleType("src.config")
    cfg.settings = mock_settings
    cfg.get_settings = lambda: mock_settings
    cfg._settings = mock_settings
    sys.modules["src.config"] = cfg

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
    from src.api.main import app
    from src.database import get_db
    from src.utils.cache import get_redis, get_redis_client
    from src.security.rate_limit import rate_limit
    from src.utils.circuit_breaker import pricing_circuit, db_circuit
    from unittest.mock import AsyncMock
    
    # Reset circuits for isolation
    if hasattr(pricing_circuit, 'reset'): pricing_circuit.reset()
    if hasattr(db_circuit, 'reset'): db_circuit.reset()

    # Mock Redis client

    mock_redis = MagicMock()

    mock_redis.get = AsyncMock(return_value=None)

    mock_redis.setex = AsyncMock(return_value=True)

    mock_redis.set = AsyncMock(return_value=True)

    mock_redis.publish = AsyncMock(return_value=1)

    

    mock_pipe = MagicMock()

    mock_pipe.execute = AsyncMock(return_value=(1, True))

    mock_redis.pipeline.return_value = mock_pipe

    

    app.dependency_overrides[get_db] = lambda: mock_db

    app.dependency_overrides[get_redis] = lambda: mock_redis

    app.dependency_overrides[get_redis_client] = lambda: mock_redis

    

    app.dependency_overrides[rate_limit] = lambda: None

    

    # 69. Test Framework: mock.patch audit logs

    with patch("src.security.audit.log_audit"), \
         patch("src.security.audit.AuditLog"), \
         patch("src.api.routes.auth._send_verification_email", new_callable=AsyncMock), \
         patch("src.api.routes.auth._send_password_reset_email", new_callable=AsyncMock), \
         patch("src.utils.cache.get_redis", return_value=mock_redis), \
         patch("src.utils.cache.init_redis_cache", new_callable=AsyncMock, return_value=mock_redis):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:

            yield ac

    # Do NOT call clear() here, isolate_dependency_overrides fixture handles it

    