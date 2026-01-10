import pytest
from unittest.mock import MagicMock, patch
from src.aiops.redis_remediator import RedisRemediator # Assuming this path

@patch("src.aiops.redis_remediator.redis.Redis")
def test_redis_remediator_init(mock_redis_class):
    """Test initialization of RedisRemediator."""
    host = "localhost"
    port = 6379
    db = 0
    remediator = RedisRemediator(host=host, port=port, db=db)
    mock_redis_class.assert_called_once_with(host=host, port=port, db=db, decode_responses=False)
    assert remediator.client is not None

@patch("src.aiops.redis_remediator.redis.Redis")
def test_redis_remediator_purge_cache_success(mock_redis_class):
    """Test successful purging of cache."""
    mock_client_instance = MagicMock()
    mock_redis_class.return_value = mock_client_instance
    
    remediator = RedisRemediator()
    cache_key_pattern = "my_cache:*"
    
    # Mocking that some keys are found and deleted
    mock_client_instance.keys.return_value = [b"my_cache:1", b"my_cache:2"]
    mock_client_instance.delete.return_value = 2 # Number of deleted keys
    
    result = remediator.purge_cache(cache_key_pattern)
    
    mock_client_instance.keys.assert_called_once_with(cache_key_pattern)
    mock_client_instance.delete.assert_called_once_with(b"my_cache:1", b"my_cache:2")
    assert result == True

@patch("src.aiops.redis_remediator.redis.Redis")
def test_redis_remediator_purge_cache_no_keys_found(mock_redis_class):
    """Test purging cache when no keys match the pattern."""
    mock_client_instance = MagicMock()
    mock_redis_class.return_value = mock_client_instance
    
    remediator = RedisRemediator()
    cache_key_pattern = "non_existent_cache:*"
    
    mock_client_instance.keys.return_value = [] # No keys found
    
    result = remediator.purge_cache(cache_key_pattern)
    
    mock_client_instance.keys.assert_called_once_with(cache_key_pattern)
    mock_client_instance.delete.assert_not_called() # Delete should not be called
    assert result == True # No keys found, so nothing to purge, still a "success" state

@patch("src.aiops.redis_remediator.redis.Redis")
def test_redis_remediator_purge_cache_failure(mock_redis_class):
    """Test purging cache when an error occurs during deletion."""
    mock_client_instance = MagicMock()
    mock_redis_class.return_value = mock_client_instance
    
    remediator = RedisRemediator()
    cache_key_pattern = "failing_cache:*"
    
    mock_client_instance.keys.return_value = [b"failing_cache:1"]
    mock_client_instance.delete.side_effect = Exception("Redis connection error")
    
    result = remediator.purge_cache(cache_key_pattern)
    
    mock_client_instance.keys.assert_called_once_with(cache_key_pattern)
    mock_client_instance.delete.assert_called_once_with(b"failing_cache:1")
    assert result == False

@patch("src.aiops.redis_remediator.redis.Redis")
def test_redis_remediator_purge_cache_empty_pattern_raises_error(mock_redis_class):
    """Test purging cache with an empty key pattern."""
    # Mock the Redis client initialization to avoid connection errors
    mock_client_instance = MagicMock()
    mock_redis_class.return_value = mock_client_instance
    
    remediator = RedisRemediator()
    with pytest.raises(ValueError, match="Cache key pattern cannot be empty."):
        remediator.purge_cache("")

@patch("src.aiops.redis_remediator.redis.Redis")
def test_redis_remediator_init_failure(mock_redis_class):
    """Test that RedisRemediator initialization raises an exception on Redis client connection failure."""
    mock_redis_class.side_effect = Exception("Redis connection failed during init")
    with pytest.raises(Exception, match="Redis connection failed during init"):
        RedisRemediator()
