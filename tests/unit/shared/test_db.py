import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.shared.db import MarketData, get_db_session, MinioStorage
from minio import Minio
import io

# Fixtures for SQLAlchemy mocks
@pytest.fixture
def mock_sqlalchemy():
    with patch('src.shared.db.create_engine') as mock_create_engine, \
         patch('src.shared.db.sessionmaker') as mock_sessionmaker:
        mock_engine = mock_create_engine.return_value
        mock_session = MagicMock()
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session)
        yield mock_create_engine, mock_sessionmaker, mock_session

# Fixtures for Minio mocks
@pytest.fixture
def mock_minio():
    with patch('src.shared.db.Minio') as mock_minio_cls:
        mock_minio_instance = mock_minio_cls.return_value
        yield mock_minio_cls, mock_minio_instance

# Test MarketData model
def test_market_data_model():
    assert MarketData.__tablename__ == 'market_data'
    assert hasattr(MarketData, 'id')
    assert hasattr(MarketData, 'ticker')
    assert hasattr(MarketData, 'timestamp')
    assert hasattr(MarketData, 'open')
    assert hasattr(MarketData, 'high')
    assert hasattr(MarketData, 'low')
    assert hasattr(MarketData, 'close')
    assert hasattr(MarketData, 'volume')

# Test get_db_session
def test_get_db_session(mock_sqlalchemy):
    mock_create_engine, mock_sessionmaker, mock_session = mock_sqlalchemy
    connection_string = "postgresql://user:password@host:port/dbname"
    
    session = get_db_session(connection_string)
    
    mock_create_engine.assert_called_once_with(connection_string)
    mock_sessionmaker.assert_called_once_with(bind=mock_create_engine.return_value)
    assert session is mock_session

# Test MinioStorage
def test_minio_storage_init(mock_minio):
    mock_minio_cls, mock_minio_instance = mock_minio
    endpoint = "minio:9000"
    access_key = "minioadmin"
    secret_key = "minioadmin"
    secure = False
    
    storage = MinioStorage(endpoint, access_key, secret_key, secure)
    
    mock_minio_cls.assert_called_once_with(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )
    assert storage.client is mock_minio_instance

def test_minio_ensure_bucket_exists(mock_minio):
    _, mock_minio_instance = mock_minio
    mock_minio_instance.bucket_exists.return_value = True
    
    storage = MinioStorage("endpoint", "ak", "sk")
    storage.ensure_bucket("test-bucket")
    
    mock_minio_instance.bucket_exists.assert_called_once_with("test-bucket")
    mock_minio_instance.make_bucket.assert_not_called()

def test_minio_ensure_bucket_not_exists(mock_minio):
    _, mock_minio_instance = mock_minio
    mock_minio_instance.bucket_exists.return_value = False
    
    storage = MinioStorage("endpoint", "ak", "sk")
    storage.ensure_bucket("new-bucket")
    
    mock_minio_instance.bucket_exists.assert_called_once_with("new-bucket")
    mock_minio_instance.make_bucket.assert_called_once_with("new-bucket")

def test_minio_upload_file(mock_minio):
    _, mock_minio_instance = mock_minio
    
    storage = MinioStorage("endpoint", "ak", "sk")
    bucket_name = "test-bucket"
    object_name = "test-object"
    data = io.BytesIO(b"test data")
    length = len(data.getvalue())
    
    storage.upload_file(bucket_name, object_name, data, length)
    
    mock_minio_instance.put_object.assert_called_once_with(
        bucket_name,
        object_name,
        data,
        length
    )

def test_minio_download_file(mock_minio):
    _, mock_minio_instance = mock_minio
    mock_return_object = MagicMock()
    mock_minio_instance.get_object.return_value = mock_return_object
    
    storage = MinioStorage("endpoint", "ak", "sk")
    bucket_name = "test-bucket"
    object_name = "test-object"
    
    result = storage.download_file(bucket_name, object_name)
    
    mock_minio_instance.get_object.assert_called_once_with(bucket_name, object_name)
    assert result is mock_return_object