# Python Backend Better Auth Integration

## Overview
Update Python backend to authenticate requests using Better Auth sessions stored in the shared Postgres database.

## Current State Analysis
- `src/auth/server.py` uses legacy `OAuth2Client` and internal JWT.
- `src/database/models.py` likely has a `User` model that might conflict or need mapping to Better Auth's schema.

## Implementation Approach
Map the Better Auth `session` and `user` tables in SQLAlchemy and create a FastAPI dependency to verify the session token.

## Phase 1: SQLAlchemy Models
### Overview
Map Better Auth tables.

### Changes Required:
#### 1. `src/database/models.py`
**Changes**: Add `BetterAuthSession` and `BetterAuthUser` (or map to existing User).
```python
from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from . import Base

class BetterAuthUser(Base):
    __tablename__ = "user" # Better Auth default
    id = Column(Text, primary_key=True)
    name = Column(Text)
    email = Column(Text, unique=True)
    emailVerified = Column(DateTime)
    image = Column(Text)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)

class BetterAuthSession(Base):
    __tablename__ = "session"
    id = Column(Text, primary_key=True)
    expiresAt = Column(DateTime)
    token = Column(Text, unique=True)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)
    ipAddress = Column(Text)
    userAgent = Column(Text)
    userId = Column(Text, ForeignKey("user.id"))
    
    user = relationship("BetterAuthUser")
```

## Phase 2: Auth Dependency
### Overview
Create `get_current_user` using the DB.

### Changes Required:
#### 1. `src/auth/better_auth.py`
**Changes**: New file.
```python
from fastapi import Request, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.database import get_db
from src.database.models import BetterAuthSession
from datetime import datetime

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    # Check Cookie
    token = request.cookies.get("better-auth.session_token")
    if not token:
        # Check Header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Verify in DB
    session = db.query(BetterAuthSession).filter(BetterAuthSession.token == token).first()
    
    if not session:
         raise HTTPException(status_code=401, detail="Invalid session")
         
    if session.expiresAt < datetime.now():
         raise HTTPException(status_code=401, detail="Session expired")
         
    return session.user
```

## Phase 3: Verify Endpoint
### Overview
Add a test endpoint.

### Changes Required:
#### 1. `src/auth/server.py`
**Changes**: Add `/auth/me` endpoint.
```python
from .better_auth import get_current_user
# ... imports

@router.get("/me")
async def read_users_me(user = Depends(get_current_user)):
    return user
```

### Success Criteria:
- [ ] Manual test: Call `/auth/me` with a valid cookie (obtained from frontend login) returns user data.
