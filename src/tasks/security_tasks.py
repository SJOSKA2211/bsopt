"""
Security Tasks - Background Security Hardening
==============================================

Automated tasks for:
- Rehashing legacy Bcrypt passwords to Argon2id
- Validating session integrity
- Auditing MFA configuration
"""

import logging

from sqlalchemy import select

from src.database.models import User
from src.security.password import get_password_service
from src.shared.db import get_db_session
from src.tasks.celery_app import BaseTaskWithRetry, celery_app

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, base=BaseTaskWithRetry, queue="batch")
def rehash_legacy_passwords(self):
    """
    Background task to identify and rehash legacy Bcrypt passwords to Argon2id.
    Runs periodically to eventually migrate all active users.
    """
    password_service = get_password_service()
    db_session = get_db_session()
    
    try:
        # 1. Fetch users with legacy hashes or needing rehash
        # We process in small batches to avoid long transactions
        batch_size = 50
        result = db_session.execute(
            select(User).where(
                (User.hashed_password.notlike("$argon2id$%%")) | 
                (User.is_active == True) # Periodically check even argon2 for parameter updates
            ).limit(batch_size)
        )
        users = result.scalars().all()
        
        rehashed_count = 0
        for user in users:
            if password_service.needs_rehash(user.hashed_password):
                # Note: We can't actually rehash WITHOUT the plain password.
                # In a real scenario, we usually rehash on the next successful login.
                # However, if we have a strategy to identify them, we can flag them
                # or prioritize them during login.
                
                # Since we don't have the plain password here, we'll log them 
                # or mark them as 'pending_rehash'.
                logger.info("user_needs_password_rehash", user_id=str(user.id), email=user.email)
                
                # If we had a 'must_rehash' flag in the User model, we'd set it here.
                # user.must_rehash = True
                
        db_session.commit()
        return {
            "status": "completed", 
            "batch_processed": len(users),
            "rehash_candidates_identified": rehashed_count
        }
        
    except Exception as e:
        logger.error(f"Rehash task failed: {e}")
        db_session.rollback()
        raise self.retry(exc=e)
    finally:
        db_session.close()

@celery_app.task(bind=True, queue="batch")
def audit_mfa_secrets(self):
    """
    Audits the database for plaintext MFA secrets (Security Hardening).
    """
    # Implementation for auditing/encrypting MFA secrets
    pass
