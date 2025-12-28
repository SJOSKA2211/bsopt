# JWT Authentication System - Setup Guide

Production-ready JWT authentication system for FastAPI with comprehensive security features.

## Overview

This authentication system provides:
- User registration and login
- JWT-based stateless authentication
- Access tokens (30 min) and refresh tokens (7 days)
- Bcrypt password hashing with configurable rounds
- Password strength validation
- Tiered rate limiting (free/pro/enterprise)
- Protected route dependencies
- Account management endpoints

## Architecture

```
src/
├── api/
│   ├── auth.py                      # Core authentication logic
│   ├── routes/
│   │   └── auth.py                  # Authentication endpoints
│   ├── schemas/
│   │   └── auth.py                  # Pydantic request/response models
│   └── middleware/
│       └── rate_limit.py            # Redis-based rate limiting
├── database/
│   ├── __init__.py                  # Database session management
│   └── models.py                    # SQLAlchemy User model
└── config.py                        # Configuration settings
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements-auth.txt
```

### 2. Database Setup

Create PostgreSQL database:

```bash
createdb options_db
```

Apply schema:

```bash
psql options_db < src/database/schema.sql
```

Or use Alembic for migrations:

```bash
alembic init alembic
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

### 3. Redis Setup

Install and start Redis:

```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### 4. Environment Variables

Create `.env` file in project root:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/options_db

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT Authentication
JWT_SECRET=your-super-secret-key-change-this-in-production-min-32-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Password Requirements
PASSWORD_MIN_LENGTH=8
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_DIGIT=true
PASSWORD_REQUIRE_SPECIAL=false
BCRYPT_ROUNDS=12

# Rate Limiting (per hour)
RATE_LIMIT_FREE=100
RATE_LIMIT_PRO=10000
RATE_LIMIT_ENTERPRISE=0

# Application
ENVIRONMENT=dev
DEBUG=true
LOG_LEVEL=INFO

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

**Security Notes:**
- Generate a strong JWT_SECRET: `openssl rand -hex 32`
- Never commit `.env` to version control
- Use different secrets for dev/staging/production
- In production, use environment variables or secrets manager

## Usage

### Starting the Application

```bash
# Development
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### API Endpoints

Base URL: `http://localhost:8000/api/v1`

#### 1. Register User

```bash
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123",
  "full_name": "John Doe"
}

# Response (201 Created)
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "user@example.com",
  "full_name": "John Doe",
  "tier": "free",
  "created_at": "2024-01-15T10:30:00",
  "last_login": null,
  "is_active": true
}
```

#### 2. Login

```bash
POST /api/v1/auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=SecurePass123

# Response (200 OK)
{
  "access_token": "<PLACEHOLDER_JWT_TOKEN>",
  "refresh_token": "<PLACEHOLDER_JWT_TOKEN>",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### 3. Get Current User

```bash
GET /api/v1/auth/me
Authorization: Bearer <access_token>

# Response (200 OK)
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "user@example.com",
  "full_name": "John Doe",
  "tier": "free",
  "created_at": "2024-01-15T10:30:00",
  "last_login": "2024-01-15T11:00:00",
  "is_active": true
}
```

#### 4. Refresh Token

```bash
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "<PLACEHOLDER_JWT_TOKEN>"
}

# Response (200 OK)
{
  "access_token": "<PLACEHOLDER_JWT_TOKEN>",
  "refresh_token": "<PLACEHOLDER_JWT_TOKEN>",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### 5. Change Password

```bash
PUT /api/v1/auth/change-password
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "old_password": "SecurePass123",
  "new_password": "NewSecurePass456"
}

# Response (200 OK)
{
  "message": "Password changed successfully",
  "detail": "Please login again with your new password"
}
```

#### 6. Logout

```bash
POST /api/v1/auth/logout
Authorization: Bearer <access_token>

# Response (200 OK)
{
  "message": "Successfully logged out",
  "detail": "Please discard your access and refresh tokens"
}
```

### Testing with cURL

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "TestPass123", "full_name": "Test User"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=TestPass123"

# Get user info (replace TOKEN with actual token)
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer TOKEN"
```

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Register
response = requests.post(
    f"{BASE_URL}/auth/register",
    json={
        "email": "user@example.com",
        "password": "SecurePass123",
        "full_name": "John Doe"
    }
)
user = response.json()
print(f"User created: {user['email']}")

# Login
response = requests.post(
    f"{BASE_URL}/auth/login",
    data={
        "username": "user@example.com",
        "password": "SecurePass123"
    }
)
tokens = response.json()
access_token = tokens["access_token"]

# Access protected endpoint
response = requests.get(
    f"{BASE_URL}/auth/me",
    headers={"Authorization": f"Bearer {access_token}"}
)
user_info = response.json()
print(f"User info: {user_info}")

# Refresh token
response = requests.post(
    f"{BASE_URL}/auth/refresh",
    json={"refresh_token": tokens["refresh_token"]}
)
new_tokens = response.json()
```

## Protecting Routes

Add authentication to any route:

```python
from fastapi import APIRouter, Depends
from src.api.auth import get_current_user
from src.database.models import User

router = APIRouter()

@router.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {
        "message": "This is a protected route",
        "user": current_user.email,
        "tier": current_user.tier
    }
```

Add rate limiting to routes:

```python
from fastapi import Depends
from src.api.middleware.rate_limit import rate_limit_dependency

@router.get(
    "/api-endpoint",
    dependencies=[Depends(rate_limit_dependency)]
)
async def rate_limited_endpoint(current_user: User = Depends(get_current_user)):
    return {"message": "This endpoint is rate limited"}
```

## Rate Limiting

Rate limits are enforced per user tier:

| Tier       | Requests/Hour | Cost      |
|------------|---------------|-----------|
| Free       | 100           | $0/month  |
| Pro        | 10,000        | $49/month |
| Enterprise | Unlimited     | Custom    |

Rate limit headers in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705320000
```

When limit is exceeded (429 response):

```json
{
  "error": "Rate limit exceeded",
  "message": "You have exceeded the rate limit for your free tier",
  "limit": 100,
  "current": 101,
  "reset": 1705320000,
  "upgrade_info": "Upgrade to a higher tier for increased rate limits"
}
```

## Security Features

### 1. Password Security
- **Bcrypt hashing** with 12 rounds (configurable)
- **Automatic salt generation** (unique per password)
- **Constant-time verification** (timing attack protection)
- **Strength validation** (uppercase, lowercase, digits)

### 2. JWT Security
- **Signed tokens** with HS256 (or RS256 for asymmetric)
- **Expiration validation** (prevents replay attacks)
- **Type verification** (distinguishes access/refresh tokens)
- **Issued-at claim** (track token age)

### 3. Authentication Security
- **Username enumeration protection** (same error for all failures)
- **Timing attack protection** (constant-time comparisons)
- **Account status checking** (inactive accounts cannot login)
- **Last login tracking** (security monitoring)

### 4. Rate Limiting
- **Token bucket algorithm** (allows traffic bursts)
- **Distributed state** (Redis for multi-server deployments)
- **Fail-open design** (Redis outage doesn't block API)
- **Per-endpoint tracking** (granular control)

### 5. Database Security
- **Parameterized queries** (SQL injection prevention via ORM)
- **Connection pooling** (resource management)
- **Unique constraints** (email uniqueness at DB level)
- **Cascade deletion** (automatic cleanup)

## Production Deployment Checklist

### Security
- [ ] Generate strong JWT_SECRET (min 32 characters)
- [ ] Use HTTPS only (set secure cookie flags)
- [ ] Enable CORS restrictions (whitelist specific origins)
- [ ] Set DEBUG=false
- [ ] Use environment variables for secrets (not .env file)
- [ ] Implement token blacklist for logout
- [ ] Add email verification for new accounts
- [ ] Implement account lockout after failed login attempts
- [ ] Add 2FA/MFA support
- [ ] Set up monitoring for suspicious login patterns

### Infrastructure
- [ ] Use managed PostgreSQL (RDS, Cloud SQL, etc.)
- [ ] Use managed Redis (ElastiCache, Cloud Memorystore, etc.)
- [ ] Set up database backups
- [ ] Configure connection pooling limits
- [ ] Set up health checks and monitoring
- [ ] Configure logging aggregation
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Implement request logging
- [ ] Configure rate limiting per IP (in addition to per user)

### Performance
- [ ] Enable database query caching
- [ ] Add database indexes (already in schema)
- [ ] Configure Redis memory limits
- [ ] Set up CDN for static assets
- [ ] Enable gzip compression
- [ ] Optimize token payload size
- [ ] Consider RS256 for better security (asymmetric keys)

### Compliance
- [ ] Implement GDPR data export
- [ ] Add password reset via email
- [ ] Implement audit logging
- [ ] Add terms of service acceptance
- [ ] Implement data retention policies
- [ ] Add user consent management

## Troubleshooting

### Database Connection Errors

```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solution:**
1. Check PostgreSQL is running: `sudo systemctl status postgresql`
2. Verify DATABASE_URL in `.env`
3. Check database exists: `psql -l`
4. Test connection: `psql <DATABASE_URL>`

### Redis Connection Errors

```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution:**
1. Check Redis is running: `redis-cli ping` (should return PONG)
2. Verify REDIS_URL in `.env`
3. Check Redis logs: `sudo journalctl -u redis`
4. Rate limiting will fail-open (allow requests) if Redis is down

### JWT Token Errors

```
HTTPException 401: Could not validate credentials
```

**Causes:**
1. Token expired (use refresh token)
2. Invalid signature (JWT_SECRET changed)
3. Malformed token
4. User deleted or deactivated

### Rate Limit Issues

```
HTTPException 429: Rate limit exceeded
```

**Solution:**
1. Wait for reset time (returned in headers)
2. Upgrade user tier
3. Clear Redis key (development only): `redis-cli DEL rate_limit:*`

## Development Tips

### Create Test User

```python
from src.database import SessionLocal
from src.database.models import User
from src.api.auth import hash_password

db = SessionLocal()
user = User(
    email="test@example.com",
    hashed_password=hash_password("TestPass123"),
    full_name="Test User",
    tier="enterprise",  # Unlimited rate limit for testing
    is_active=True
)
db.add(user)
db.commit()
```

### Decode JWT Token (for debugging)

```python
from src.api.auth import decode_token

token = "<PLACEHOLDER_JWT_TOKEN>"
payload = decode_token(token)
print(payload)
```

### Check Rate Limit Status

```python
import redis

r = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
keys = r.keys("rate_limit:*")
for key in keys:
    count = r.get(key)
    ttl = r.ttl(key)
    print(f"{key}: {count} requests, expires in {ttl}s")
```

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Support

For issues or questions:
1. Check logs: `tail -f logs/app.log`
2. Enable debug mode: `DEBUG=true` in `.env`
3. Review FastAPI docs: https://fastapi.tiangolo.com
4. Review JWT docs: https://pyjwt.readthedocs.io

## License

This authentication system is part of the Options Pricing Platform.
