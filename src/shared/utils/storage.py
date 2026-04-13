from typing import BinaryIO

import boto3
import structlog
from botocore.client import Config

logger = structlog.get_logger()


class ObjectStorageManager:
    """
    Production-grade Object Storage Manager (S3/MinIO).
    Handles artifacts, logs, and raw data persistence.
    """

    def __init__(self):
        from src.shared.config import settings

        self.endpoint = settings.MINIO_ENDPOINT
        self.access_key = settings.MINIO_ROOT_USER
        self.secret_key = settings.MINIO_ROOT_PASSWORD
        self.use_ssl = settings.MINIO_USE_SSL

        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.MINIO_ENDPOINT_URL,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def upload_file(self, bucket: str, object_name: str, file_path: str):
        """Upload a file to the specified bucket."""
        try:
            self.s3.upload_file(file_path, bucket, object_name)
            logger.info("file_upload_success", bucket=bucket, object=object_name)
        except Exception as e:
            logger.error("file_upload_failed", bucket=bucket, object=object_name, error=str(e))
            raise

    def upload_fileobj(self, bucket: str, object_name: str, data: BinaryIO):
        """Upload a file-like object."""
        try:
            self.s3.upload_fileobj(data, bucket, object_name)
            logger.info("fileobj_upload_success", bucket=bucket, object=object_name)
        except Exception as e:
            logger.error("fileobj_upload_failed", bucket=bucket, object=object_name, error=str(e))
            raise

    def download_file(self, bucket: str, object_name: str, file_path: str):
        """Download a file from an S3 bucket."""
        try:
            self.s3.download_file(bucket, object_name, file_path)
            logger.info("file_download_success", bucket=bucket, object=object_name)
        except Exception as e:
            logger.error("file_download_failed", bucket=bucket, object=object_name, error=str(e))
            raise

    def get_presigned_url(self, bucket: str, object_name: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for secure temporary access."""
        try:
            url = self.s3.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": object_name}, ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error("presigned_url_generation_failed", object=object_name, error=str(e))
            raise


storage_manager = ObjectStorageManager()