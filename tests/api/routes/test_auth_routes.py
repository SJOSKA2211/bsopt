import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.auth.auth import auth_service
from src.auth.passwords import password_service
from src.database import get_db
from src.database.models import User
from src.shared.config import settings

client = TestClient(app)

@pytest.fixture
def test_user(api_client):
    """Creates a real test user in the database."""
    email = f"test_{uuid.uuid4().hex[:8]}@Manifold.io"
    password = "ProductionPassword123!"
    hashed = password_service.hash_password(password)
    
    # We use a raw DB session for clean setup
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine(settings.DATABASE_URL.replace("+asyncpg", ""))
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    user = User(
        email=email,
        hashed_password=hashed,
        full_name="Production Tester",
        is_verified=True,
        is_active=True,
        tier="pro"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    
    return {"email": email, "password": password, "user": user}

def test_registration_and_login_flow(api_client):
    """Verified End-to-End Auth Flow (Data-Driven)."""
    email = f"new_{uuid.uuid4().hex[:8]}@Manifold.io"
    payload = {
        "email": email,
        "password": "ProductionPassword123!",
        "password_confirm": "ProductionPassword123!",
        "full_name": "New Production User",
        "accept_terms": True,
    }
    
    # 1. Register
    resp = api_client.post("/api/v1/auth/register", json=payload)
    assert resp.status_code == 201, resp.text
    
    # 2. Login
    login_resp = api_client.post(
        "/api/v1/auth/login", 
        json={"email": email, "password": payload["password"]}
    )
    assert login_resp.status_code == 200
    data = login_resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_password(api_client, test_user):
    """Production rejection of invalid credentials."""
    resp = api_client.post(
        "/api/v1/auth/login",
        json={"email": test_user["email"], "password": "WrongPassword!"}
    )
    # Depending on implementation, might be 401 or 400
    assert resp.status_code in [400, 401]

def test_protected_route_access(api_client, test_user):
    """Verify JWT enforcement and data-driven access."""
    # 1. Get Token
    login_resp = api_client.post(
        "/api/v1/auth/login", 
        json={"email": test_user["email"], "password": test_user["password"]}
    )
    token = login_resp.json()["access_token"]
    
    # 2. Access protected route (e.g., current user info)
    resp = api_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == test_user["email"]

def test_logout_and_token_invalidation(api_client, test_user):
    """Production cleanup and token lifecycle verification."""
    # 1. Login
    login_resp = api_client.post(
        "/api/v1/auth/login", 
        json={"email": test_user["email"], "password": test_user["password"]}
    )
    token = login_resp.json()["access_token"]
    
    # 2. Logout
    logout_resp = api_client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert logout_resp.status_code == 200
    
    # 3. Verify token is now invalid
    retry_resp = api_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert retry_resp.status_code == 401
