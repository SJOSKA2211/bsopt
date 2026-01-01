import time
from typing import Dict, Optional, Tuple
from fastapi import HTTPException, Request, status
from src.config import settings

class RateLimiter:
    """
    Simple in-memory rate limiter.
    In production, this should use Redis for distributed rate limiting.
    """
    def __init__(self):
        # bucket: {ip/user_id: [timestamps]}
        self.buckets: Dict[str, list] = {}

    def is_allowed(self, identifier: str, limit: int, window: int = 60) -> Tuple[bool, int]:
        """
        Check if request is allowed under rate limit.
        Returns (is_allowed, remaining_requests).
        """
        if limit == 0:  # 0 means unlimited
            return True, 999999
            
        now = time.time()
        if identifier not in self.buckets:
            self.buckets[identifier] = []
            
        # Clean up old timestamps
        self.buckets[identifier] = [t for d, t in enumerate(self.buckets[identifier]) if t > now - window]
        
        if len(self.buckets[identifier]) < limit:
            self.buckets[identifier].append(now)
            return True, limit - len(self.buckets[identifier])
            
        return False, 0

limiter = RateLimiter()

async def rate_limit(request: Request):
    """
    Rate limiting dependency.
    """
    # Prefer user_id if authenticated, otherwise use IP
    user = getattr(request.state, "user", None)
    identifier = str(user.id) if user else request.client.host
    tier = getattr(user, "tier", "free") if user else "free"
    
    limit = settings.rate_limit_tiers.get(tier, 100)
    
    allowed, remaining = limiter.is_allowed(identifier, limit)
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Upgrade your tier for higher limits.",
            headers={"Retry-After": "60"}
        )
    
    # We could add X-RateLimit-Remaining header here if we want
    return remaining
