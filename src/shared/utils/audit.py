import os
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class InstitutionalAuditLog:
    """
    Centralized Institutional Audit Log Service.
    Ensures all critical actions are recorded with high fidelity and tamper-evident metadata.
    """

    def __init__(self):
        self.service_name = os.getenv("SERVICE_NAME", "EquaFlow")
        self.log_target = os.getenv("AUDIT_LOG_TARGET", "database")  # 'database' or 'stdout'

    def log_action(self, actor_id: str, action: str, metadata: dict[str, Any]):
        """
        Log a critical institutional action.
        """
        timestamp = time.time()
        import asyncio
        import hashlib
        import hmac
        import json

        from sqlalchemy import text

        from src.shared.config import settings

        payload = json.dumps({
            "timestamp": timestamp,
            "actor_id": actor_id,
            "action": action,
            "service": self.service_name,
            "metadata": metadata
        }, sort_keys=True)
        
        secret = settings.SECRET_KEY.encode() if settings.SECRET_KEY else b"default_audit_secret"
        signature = hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()

        audit_entry = {
            "timestamp": timestamp,
            "actor_id": actor_id,
            "action": action,
            "service": self.service_name,
            "metadata": metadata,
            "integrity_hash": f"sha256:{signature}",
        }

        # 1. Structured Logging (Standard)
        logger.info("institutional_audit_event", **audit_entry)

        # 2. Database Persistence (Institutional Compliance)
        if self.log_target == "database":
            async def _persist_audit(entry: dict):
                try:
                    from src.database import db_manager
                    query = text("""
                        INSERT INTO institutional_audit_logs 
                        (timestamp, actor_id, action, service, metadata, integrity_hash)
                        VALUES (to_timestamp(:timestamp), :actor_id, :action, :service, :metadata_json, :integrity_hash)
                    """)
                    async with db_manager.async_engine.begin() as conn:
                        await conn.execute(query, {
                            "timestamp": entry["timestamp"],
                            "actor_id": entry["actor_id"],
                            "action": entry["action"],
                            "service": entry["service"],
                            "metadata_json": json.dumps(entry["metadata"]),
                            "integrity_hash": entry["integrity_hash"]
                        })
                except Exception as e:
                    logger.error("audit_database_persistence_failed", error=str(e))
            
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_persist_audit(audit_entry))
            except RuntimeError:
                logger.warning("no_running_event_loop_for_audit_db_persistence")


audit_logger = InstitutionalAuditLog()
