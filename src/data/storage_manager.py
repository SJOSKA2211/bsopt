import asyncio
from typing import Any

import aioboto3
import structlog
from botocore.exceptions import ClientError

logger = structlog.get_logger(__name__)


class AsyncStorageManager:
    """
    High-performance asynchronous S3/MinIO storage manager.
    OPTIMIZED: Concurrency-aware background tasking.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        region: str = "us-east-1",
        max_concurrent_uploads: int = 5,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.session = aioboto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_uploads)

    async def upload_file(self, local_path: str, remote_key: str) -> None:
        """Asynchronous file upload with semaphore protection."""
        async with self._semaphore:
            async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
                try:
                    logger.info("uploading_to_vault", key=remote_key)
                    await s3.upload_file(local_path, self.bucket_name, remote_key)
                    logger.info("upload_to_vault_success", key=remote_key)
                except ClientError as e:
                    logger.error("upload_to_vault_failed", key=remote_key, error=str(e))
                    raise

    async def download_file(self, remote_key: str, local_path: str) -> None:
        """Asynchronous file download."""
        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            try:
                logger.info("downloading_from_vault", key=remote_key)
                await s3.download_file(self.bucket_name, remote_key, local_path)
                logger.info("download_from_vault_success", key=remote_key)
            except ClientError as e:
                logger.error("download_from_vault_failed", key=remote_key, error=str(e))
                raise

    def upload_background(self, local_path: str, remote_key: str) -> asyncio.Task[None]:
        """Fire-and-forget background upload (non-blocking)."""
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.upload_file(local_path, remote_key))
        # Optional: Track tasks if needed for coordination
        return task


# Example usage (Mock)
async def test_storage() -> None:
    AsyncStorageManager("http://localhost:9000", "admin", "password", "models")
    # await manager.upload_file("model.pt", "v1/model.pt")
