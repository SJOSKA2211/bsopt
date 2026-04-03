import asyncio
import logging
from datetime import datetime

from sqlalchemy import update

from src.database import db_manager
from src.database.models import APIKey
from src.shared.utils.cache import get_redis

logger = logging.getLogger("auth_tasks")


async def flush_api_key_usage_loop():
    """Background task to flush buffered API key usage timestamps to the database."""
    logger.info("🔄 API Key usage flush loop started")
    while True:
        try:
            await asyncio.sleep(60)  # Flush every minute
            redis = get_redis()
            if not redis:
                continue

            # Get all buffered updates
            updates = await redis.hgetall("api_key_last_used")
            if not updates:
                continue

            logger.info(f"💾 Flushing {len(updates)} API key usage updates to database...")

            async with db_manager.async_session_factory() as db:
                for key_hash_raw, last_used_raw in updates.items():
                    key_hash = (
                        key_hash_raw.decode() if isinstance(key_hash_raw, bytes) else key_hash_raw
                    )
                    last_used_str = (
                        last_used_raw.decode()
                        if isinstance(last_used_raw, bytes)
                        else last_used_raw
                    )

                    last_used = datetime.fromisoformat(last_used_str)

                    await db.execute(
                        update(APIKey)
                        .where(APIKey.key_hash == key_hash)
                        .values(last_used_at=last_used)
                    )
                await db.commit()

            # Clear flushed updates from Redis
            # Use a transaction-safe approach? For simplicity we just delete the hash
            # but in production we might want to only delete what we processed
            await redis.delete("api_key_last_used")
            logger.info("✅ API Key usage flush complete")

        except asyncio.CancelledError:
            logger.info("🛑 API Key usage flush loop stopping...")
            break
        except Exception as e:
            logger.error("api_key_usage_flush_failed", error=str(e))
