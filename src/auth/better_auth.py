from datetime import datetime

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.database import get_db
from src.database.models import BetterAuthSession, User  # Import legacy User


async def get_current_user(request: Request, db: Session = Depends(get_db)):
    # Check Cookie
    token = request.cookies.get("better-auth.session_token")
    if not token:
        # Check Header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    if not token:
        return None  # Return None instead of raising so flexible auth can fallback

    # Verify in DB
    session = (
        db.query(BetterAuthSession).filter(BetterAuthSession.token == token).first()
    )

    if not session:
        return None

    if session.expiresAt < datetime.now():
        return None

    # 🚀 SINGULARITY: Link Better Auth user to Legacy User via Email
    # This ensures all existing fields (tier, portfolios) are available
    user = db.query(User).filter(User.email == session.user.email).first()

    if not user:
        # If not found, we could auto-create a legacy User record here
        # For now, we'll just use the session.user but this might cause issues
        # with code expecting 'tier'.
        # Optimization: Return session.user as fallback
        return session.user

    return user
