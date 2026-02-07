from unittest.mock import MagicMock, patch

import pytest

from src.aiops.redis_remediator import RedisRemediator


@patch("src.aiops.redis_remediator.logger")
@patch("src.aiops.redis_remediator.redis.Redis")
class TestRedisRemediator:
    def test_redis_remediator_init_success(self, mock_redis_class, mock_logger):
        """Test successful initialization of RedisRemediator."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        host = "localhost"
        port = 6379
        db = 0
        remediator = RedisRemediator(host=host, port=port, db=db)
        mock_redis_class.assert_called_once_with(host=host, port=port, db=db, decode_responses=False)
        mock_redis_instance.ping.assert_called_once()
        assert remediator.client is not None
        mock_logger.info.assert_called_once_with(
            "redis_remediator_init", status="success", message="Redis client initialized and connected."
        )
        mock_logger.error.assert_not_called()

    def test_redis_remediator_init_failure(self, mock_redis_class, mock_logger):
        """Test that RedisRemediator initialization raises an exception on Redis client connection failure."""
        mock_redis_class.side_effect = Exception("Redis connection failed during init")
        with pytest.raises(Exception, match="Redis connection failed during init"):
            RedisRemediator()
        mock_logger.error.assert_called_once_with(
            "redis_remediator_init", status="failure", error="Redis connection failed during init", message="Failed to initialize or connect to Redis client."
        )
        mock_logger.info.assert_not_called()
        # No ping in this path
        # mock_redis_instance.ping.assert_not_called() # This mock is not available here, handled by side_effect

    def test_redis_remediator_purge_cache_success(self, mock_redis_class, mock_logger):
        """Test successful purging of cache."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True # For init to succeed
        
        remediator = RedisRemediator()
        mock_logger.reset_mock() # Reset logger calls after init
        cache_key_pattern = "my_cache:*"
        
        # Mocking that some keys are found and deleted
        mock_redis_instance.keys.return_value = [b"my_cache:1", b"my_cache:2"]
        mock_redis_instance.delete.return_value = 2 # Number of deleted keys
        
        result = remediator.purge_cache(cache_key_pattern)
        
        mock_redis_instance.keys.assert_called_once_with(cache_key_pattern)
        mock_redis_instance.delete.assert_called_once_with(b"my_cache:1", b"my_cache:2")
        assert result
        mock_logger.info.assert_called_once_with(
            "redis_remediator_purge", status="success", pattern=cache_key_pattern, deleted_count=2, message="Cache purged successfully."
        )
        mock_logger.error.assert_not_called()

    def test_redis_remediator_purge_cache_no_keys_found(self, mock_redis_class, mock_logger):
        """Test purging cache when no keys match the pattern."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True # For init to succeed

        remediator = RedisRemediator()
        mock_logger.reset_mock() # Reset logger calls after init
        cache_key_pattern = "non_existent_cache:*"
        
        mock_redis_instance.keys.return_value = [] # No keys found
        
        result = remediator.purge_cache(cache_key_pattern)
        
        mock_redis_instance.keys.assert_called_once_with(cache_key_pattern)
        mock_redis_instance.delete.assert_not_called() # Delete should not be called
        assert result # No keys found, so nothing to purge, still a "success" state
        mock_logger.info.assert_called_once_with(
            "redis_remediator_purge", status="no_keys_found", pattern=cache_key_pattern, message="No keys matching pattern found to purge."
        )
        mock_logger.error.assert_not_called()

    def test_redis_remediator_purge_cache_failure(self, mock_redis_class, mock_logger):
        """Test purging cache when an error occurs during deletion."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True # For init to succeed

        remediator = RedisRemediator()
        mock_logger.reset_mock() # Reset logger calls after init
        cache_key_pattern = "failing_cache:*"
        
        mock_redis_instance.keys.return_value = [b"failing_cache:1"]
        mock_redis_instance.delete.side_effect = Exception("Redis connection error")
        
        result = remediator.purge_cache(cache_key_pattern)
        
        mock_redis_instance.keys.assert_called_once_with(cache_key_pattern)
        mock_redis_instance.delete.assert_called_once_with(b"failing_cache:1")
        assert not result
        mock_logger.error.assert_called_once_with(
            "redis_remediator_purge", status="failure", pattern=cache_key_pattern, error="Redis connection error", message="Failed to purge cache."
        )
        mock_logger.info.assert_not_called()

    def test_redis_remediator_purge_cache_empty_pattern_raises_error(self, mock_redis_class, mock_logger):
        """Test purging cache with an empty key pattern."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True # For init to succeed
        
        remediator = RedisRemediator()
        mock_logger.reset_mock() # Reset logger calls after init
        with pytest.raises(ValueError, match="Cache key pattern cannot be empty."):
            remediator.purge_cache("")
        mock_logger.info.assert_not_called()
        mock_logger.error.assert_not_called()
        mock_logger.warning.assert_not_called()
