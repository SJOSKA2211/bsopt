from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger
from sqlalchemy.orm import sessionmaker, declarative_base
from minio import Minio
import io

Base = declarative_base()

class MarketData(Base):
    """
    SQLAlchemy model for market data stored in TimescaleDB.
    """
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False, index=True)
    timestamp = Column(BigInteger, nullable=False, index=True) # Unix timestamp in milliseconds
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

def get_db_session(connection_string: str):
    """
    Creates a new database session.
    
    Args:
        connection_string: SQLAlchemy connection string.
        
    Returns:
        SQLAlchemy Session object.
    """
    engine = create_engine(connection_string)
    Session = sessionmaker(bind=engine)
    return Session()

class MinioStorage:
    """
    Client for MinIO storage for cold data and artifacts.
    """
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def ensure_bucket(self, bucket_name: str):
        """Ensures that the bucket exists, creating it if necessary."""
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    def upload_file(self, bucket_name: str, object_name: str, data: io.BytesIO, length: int):
        """Uploads a file to the specified bucket."""
        self.client.put_object(bucket_name, object_name, data, length)

    def download_file(self, bucket_name: str, object_name: str):
        """Downloads a file from the specified bucket."""
        return self.client.get_object(bucket_name, object_name)
