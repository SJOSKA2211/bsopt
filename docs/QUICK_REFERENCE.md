# JWT Authentication - Quick Reference Card

## Setup (5 Minutes)

```bash
# 1. Run setup script
./setup_auth.sh

# 2. Update .env file (optional)
nano .env

# 3. Start server
uvicorn src.api.main:app --reload

# 4. Test authentication
python test_auth.py
```

## API Endpoints

### Register User
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123","full_name":"John Doe"}'
```

### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=user@example.com&password=SecurePass123"
```

### Get Current User (Protected)
```bash
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Refresh Token
```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"YOUR_REFRESH_TOKEN"}'
```

### Change Password
```bash
curl -X PUT http://localhost:8000/api/v1/auth/change-password \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"old_password":"OldPass123","new_password":"NewPass456"}'
```

## Python Usage

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Register
response = requests.post(f"{BASE_URL}/auth/register", json={
    "email": "user@example.com",
    "password": "SecurePass123",
    "full_name": "John Doe"
})

# Login
response = requests.post(f"{BASE_URL}/auth/login", data={
    "username": "user@example.com",
    "password": "SecurePass123"
})
tokens = response.json()

# Access protected endpoint
response = requests.get(
    f"{BASE_URL}/auth/me",
    headers={"Authorization": f"Bearer {tokens['access_token']}"}
)
```

## FastAPI Route Protection

```python
from fastapi import Depends
from src.api.auth import get_current_user
from src.database.models import User

@app.get("/protected")
async def protected_route(user: User = Depends(get_current_user)):
    return {"message": f"Hello {user.email}"}
```

## Rate Limiting

```python
from fastapi import Depends
from src.api.middleware.rate_limit import rate_limit_dependency

@app.get("/limited", dependencies=[Depends(rate_limit_dependency)])
async def limited_route(user: User = Depends(get_current_user)):
    return {"message": "Rate limited endpoint"}
```

## Configuration (.env)

```env
# Required
DATABASE_URL=postgresql://user:password@localhost:5432/options_db
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=your-32-character-secret-key-here

# Optional (defaults shown)
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
RATE_LIMIT_FREE=100
RATE_LIMIT_PRO=10000
RATE_LIMIT_ENTERPRISE=0
```

## Rate Limits

| Tier | Limit | Cost |
|------|-------|------|
| Free | 100/hour | Free |
| Pro | 10K/hour | $49/mo |
| Enterprise | Unlimited | Custom |

## Common Errors

### 401 Unauthorized
- Invalid or expired token
- User not found or inactive
- **Fix:** Login again or refresh token

### 422 Validation Error
- Invalid email format
- Weak password
- **Fix:** Check password requirements (min 8 chars, uppercase, lowercase, digit)

### 429 Rate Limit Exceeded
- Too many requests
- **Fix:** Wait for reset time or upgrade tier

### 500 Database Error
- Database not running
- **Fix:** Start PostgreSQL: `sudo systemctl start postgresql`

### Redis Connection Error
- Redis not running (rate limiting disabled)
- **Fix:** Start Redis: `sudo systemctl start redis`

## Password Requirements

- Minimum 8 characters
- At least one uppercase letter (A-Z)
- At least one lowercase letter (a-z)
- At least one digit (0-9)
- Special characters optional

## Token Information

**Access Token:**
- Lifetime: 30 minutes
- Use: Authenticate API requests
- Header: `Authorization: Bearer <token>`

**Refresh Token:**
- Lifetime: 7 days
- Use: Get new access tokens
- Endpoint: `POST /auth/refresh`

## Testing

```bash
# Run automated tests
python test_auth.py

# Manual health check
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

## File Locations

```
src/api/auth.py              # Core authentication logic
src/api/routes/auth.py       # API endpoints
src/api/schemas/auth.py      # Request/response models
src/api/middleware/rate_limit.py  # Rate limiting
src/database/models.py       # User model
src/config.py                # Configuration
```

## Useful Commands

```bash
# Start server
uvicorn src.api.main:app --reload

# Generate JWT secret
openssl rand -hex 32

# Check Redis
redis-cli ping

# Check PostgreSQL
psql -l

# View logs
tail -f logs/app.log

# Create database
createdb options_db

# Apply schema
psql options_db < src/database/schema.sql
```

## Documentation

- **Setup Guide:** AUTH_SETUP.md
- **Full Summary:** AUTH_IMPLEMENTATION_SUMMARY.md
- **API Docs:** http://localhost:8000/docs

## Security Checklist

Development:
- [x] JWT secret generated
- [x] Database connection secured
- [x] Password hashing enabled
- [x] Rate limiting configured

Production:
- [ ] HTTPS enabled
- [ ] JWT secret rotated
- [ ] CORS configured
- [ ] DEBUG=false
- [ ] Monitoring enabled
- [ ] Email verification
- [ ] Token blacklist
- [ ] Account lockout

## Support

1. Check AUTH_SETUP.md troubleshooting
2. Run test_auth.py
3. Enable DEBUG=true
4. Check server logs
5. Verify Redis and PostgreSQL running
