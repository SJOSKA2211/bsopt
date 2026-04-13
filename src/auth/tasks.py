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
    logger.info("api_key_usage_flush_loop_started")
    while True:
        try:
            await asyncio.sleep(60)  # Flush every minute
            redis = get_redis()
            if not redis:
                continue

            # Atomic transition to processing key
            processing_key = f"api_key_last_used:processing:{datetime.now().timestamp()}"
            if await redis.rename("api_key_last_used", processing_key):
                # Get all buffered updates from the processing key
                updates = await redis.hgetall(processing_key)
                if not updates:
                    await redis.delete(processing_key)
                    continue

                logger.info("flushing_api_key_usage_updates", count=len(updates))

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

                # Safely delete the processed key
                await redis.delete(processing_key)
                logger.info("api_key_usage_flush_complete")

        except asyncio.CancelledError:
            logger.info("api_key_usage_flush_loop_stopping")
            break
        except Exception as e:
            logger.error("api_key_usage_flush_failed", error=str(e))