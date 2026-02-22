"""
Storage Manager
"""

import asyncio
import os


class StorageManager:
    """
    Unified Async Storage Interface.
    Supports S3 (MinIO) and Local File System.
    """

    def __init__(
        self,
        provider: str = "s3",
        bucket: str = "bsopt-data",
        endpoint_url: str | None = None,
        max_concurrent_uploads: int = 10,
    ):
        self.provider = provider
        self.bucket = bucket
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        self._semaphore = asyncio.Semaphore(max_concurrent_uploads)

        if self.provider == "s3":
            import boto3

            # Note: boto3 is synchronous. For high-performance async S3, we use aiobotocore.
            # But here we assume a session wrapper or sync-to-async adapter.
            self.session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            )

    async def upload_file(self, local_path: str, remote_key: str):
        """Asynchronous file upload with semaphore protection."""
        async with self._semaphore:
            if self.provider == "local":
                import shutil

                dest_path = os.path.join(self.bucket, remote_key)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                await asyncio.to_thread(shutil.copy2, local_path, dest_path)
            elif self.provider == "s3":
                # For demonstration, wrapping blocking call
                # Ideally: use aiobotocore
                def _upload():
                    s3 = self.session.client("s3", endpoint_url=self.endpoint_url)
                    try:
                        s3.upload_file(local_path, self.bucket, remote_key)
                    except Exception:
                        # Auto-create bucket if missing (dev convenience)
                        try:
                            s3.create_bucket(Bucket=self.bucket)
                            s3.upload_file(local_path, self.bucket, remote_key)
                        except Exception:
                            raise

                await asyncio.to_thread(_upload)
