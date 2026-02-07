import io

from minio import Minio
from sqlalchemy.orm import Session

from src.database import get_session


def get_db_session(connection_string: str = None) -> Session:
    """
    Creates a new database session using the central optimized engine.
    """
    return get_session()


class MinioStorage:
    """
    Client for MinIO storage for cold data and artifacts.
    """

    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, secure: bool = False
    ):
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )

    def ensure_bucket(self, bucket_name: str):
        """Ensures that the bucket exists, creating it if necessary."""
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    def upload_file(
        self, bucket_name: str, object_name: str, data: io.BytesIO, length: int
    ):
        """Uploads a file to the specified bucket."""
        self.client.put_object(bucket_name, object_name, data, length)

    def download_file(self, bucket_name: str, object_name: str):
        """Downloads a file from the specified bucket."""
        return self.client.get_object(bucket_name, object_name)
