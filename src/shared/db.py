import io
from typing import Any

from anyio.to_thread import run_sync
from minio import Minio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.database import AsyncSessionLocal, SessionLocal


def get_db_session() -> Session:
    """Creates a new sync database session (use with caution)."""
    return SessionLocal()


async def get_async_db_session() -> AsyncSession:
    """Creates a new high-performance async database session."""
    return AsyncSessionLocal()


class MinioStorage:
    """
    Client for MinIO storage with OPTIMIZED non-blocking execution.
    """

    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, secure: bool = False
    ):
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )

    async def ensure_bucket(self, bucket_name: str):
        """Ensures that the bucket exists (Non-blocking)."""
        await run_sync(self._ensure_bucket_sync, bucket_name)

    def _ensure_bucket_sync(self, bucket_name: str):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    async def upload_file(
        self, bucket_name: str, object_name: str, data: io.BytesIO, length: int
    ):
        """Uploads a file (Non-blocking)."""
        await run_sync(self.client.put_object, bucket_name, object_name, data, length)

    async def download_file(self, bucket_name: str, object_name: str) -> Any:
        """Downloads a file (Non-blocking)."""
        return await run_sync(self.client.get_object, bucket_name, object_name)
