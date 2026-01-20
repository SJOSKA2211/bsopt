import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
import asyncio
import logging
from datetime import datetime, timedelta
import os
import pandas as pd
from pathlib import Path

from src.tasks.data_tasks import (
    collect_options_data_task,
    validate_collected_data_task,
    check_data_freshness_task,
    scheduled_data_collection,
    run_full_data_pipeline_task,
)

# Mock the module-level logger
@pytest.fixture(autouse=True)
def mock_data_tasks_logger():
    with patch('src.tasks.data_tasks.logger') as mock_logger:
        yield mock_logger



@pytest.fixture
def mock_pipeline_dependencies():
    with patch('src.data.pipeline.DataPipeline') as MockDataPipeline, \
         patch('src.data.pipeline.PipelineConfig') as MockPipelineConfig, \
         patch('src.data.pipeline.StorageBackend') as MockStorageBackend, \
         patch('src.tasks.data_tasks.asyncio.new_event_loop') as mock_new_event_loop, \
         patch('src.tasks.data_tasks.asyncio.set_event_loop') as mock_set_event_loop, \
         patch('os.path.getmtime') as mock_getmtime, \
         patch('src.tasks.data_tasks.datetime') as MockDatetime:
        
        mock_pipeline_instance = MockDataPipeline.return_value
        mock_pipeline_instance.run = AsyncMock(return_value={
            "samples_collected": 50000,
            "samples_valid": 49000,
            "output_path": "/mock/data/pipeline_1",
            "duration_seconds": 60.0,
            "validation_rate": 0.98,
        })

        mock_loop_instance = MagicMock()
        mock_new_event_loop.return_value = mock_loop_instance

        # Configure datetime mock
        mock_now = MagicMock()
        mock_now.timestamp.return_value = datetime.now().timestamp()
        MockDatetime.now.return_value = mock_now
        MockDatetime.side_effect = lambda *args, **kw: datetime(*args, **kw) # Allow real datetime for timedelta
        
        yield (MockDataPipeline, MockPipelineConfig, MockStorageBackend, mock_pipeline_instance, 
               mock_new_event_loop, mock_loop_instance, mock_getmtime, MockDatetime)

# --- Tests for collect_options_data_task ---
def test_collect_options_data_task_success(mock_pipeline_dependencies, mock_data_tasks_logger):
    (MockDataPipeline, MockPipelineConfig, MockStorageBackend, mock_pipeline_instance, 
     mock_new_event_loop, mock_loop_instance, mock_getmtime, MockDatetime) = mock_pipeline_dependencies
    
    with patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        mock_request.id = "test_task_id"
        # Ensure that the mock_loop_instance returns a value that can be used by the logger's f-string
        mock_loop_instance.run_until_complete.return_value = {
            "samples_collected": 50000,
            "samples_valid": 49000,
            "output_path": "/mock/data/pipeline_1",
            "duration_seconds": 60.0,
            "validation_rate": 0.98,
        }
        # Since collect_options_data_task is not an async function itself,
        # we call it directly. Its internal call to pipeline.run() is awaited by asyncio.run_until_complete.
        result = collect_options_data_task.run()
    
    MockPipelineConfig.assert_called_once()
    MockDataPipeline.assert_called_once_with(MockPipelineConfig.return_value)
    mock_pipeline_instance.run.assert_called_once() # This should now pass, as the internal async function is awaited
    mock_new_event_loop.assert_called_once()
    mock_loop_instance.run_until_complete.assert_called_once()
    mock_loop_instance.close.assert_called_once()
    mock_data_tasks_logger.info.assert_any_call("Starting options data collection for default symbols")
    mock_data_tasks_logger.info.assert_any_call("Data collection completed: 49000 samples")
    mock_data_tasks_logger.error.assert_not_called()

    assert result["status"] == "success"
    assert result["samples_valid"] == 49000

def test_collect_options_data_task_exception(mock_pipeline_dependencies, mock_data_tasks_logger):
    (MockDataPipeline, _, _, mock_pipeline_instance, 
     mock_new_event_loop, mock_loop_instance, _, _) = mock_pipeline_dependencies # Unpack more mocks
    
    with patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        mock_request.id = "test_task_id"
        # Mock loop.run_until_complete to raise the exception, as it's the one catching it
        mock_loop_instance.run_until_complete.side_effect = Exception("Pipeline failed")
        
        with pytest.raises(Exception, match="Pipeline failed"):
            # Since collect_options_data_task is not an async function itself,
            # we call it directly. Its internal call to pipeline.run() is awaited by asyncio.run_until_complete.
            collect_options_data_task.run(symbols=["TEST"])
    
    mock_data_tasks_logger.error.assert_called_once_with("Data collection failed: Pipeline failed")

# --- Tests for validate_collected_data_task ---
def test_validate_collected_data_task_success(mock_pipeline_dependencies, mock_data_tasks_logger):
    (_, _, _, _, _, _, _, _) = mock_pipeline_dependencies
    
    with patch('pathlib.Path') as MockPath, \
         patch('pandas.read_parquet') as mock_read_parquet, \
                      patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request: # Patch request        
        mock_data_path = "/mock/data/pipeline_1"
        
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.__truediv__.return_value.exists.return_value = True # for parquet_path.exists()
        
        mock_df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "target": [10, 11, 12, 13, 14]
        })
        # Mock pandas.read_parquet directly since it's imported as pd
        mock_read_parquet.return_value = mock_df
        mock_request.id = "test_task_id" # Set request ID
        
        result = validate_collected_data_task.run(mock_data_path)
        
        MockPath.assert_called_once_with(mock_data_path)
        mock_path_instance.__truediv__.assert_called_once_with("training_data.parquet")
        mock_read_parquet.assert_called_once_with(mock_path_instance.__truediv__.return_value)
        mock_data_tasks_logger.info.assert_any_call(f"Validating data at {mock_data_path}")
        mock_data_tasks_logger.info.assert_any_call(
            "Validation complete: quality_score=1.000, passed=True"
        )
        assert result["status"] == "success"
        assert result["validation"]["passed"] is True
        assert result["validation"]["quality_score"] == 1.0

import re # Import re for re.escape

def test_validate_collected_data_task_file_not_found(mock_pipeline_dependencies, mock_data_tasks_logger):
    (_, _, _, _, _, _, _, _) = mock_pipeline_dependencies
    
    with patch('pathlib.Path') as MockPath, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        mock_data_path = "/mock/data/pipeline_1"
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True # for data_dir.exists()
        mock_path_instance.__truediv__.return_value.exists.return_value = False # for parquet_path.exists()
        mock_request.id = "test_task_id"
        
        # Use re.escape to handle MagicMock in the exception message
        expected_error_message = re.escape(f"No parquet file found at {mock_path_instance.__truediv__.return_value}")
        with pytest.raises(FileNotFoundError, match=expected_error_message):
            validate_collected_data_task.run(mock_data_path)
        
        mock_data_tasks_logger.error.assert_called_once_with(f"Data validation failed: No parquet file found at {mock_path_instance.__truediv__.return_value}")

def test_validate_collected_data_task_exception(mock_pipeline_dependencies, mock_data_tasks_logger):
    (_, _, _, _, _, _, _, _) = mock_pipeline_dependencies
    
    with patch('pathlib.Path') as MockPath, \
         patch('pandas.read_parquet') as mock_read_parquet, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        mock_data_path = "/mock/data/pipeline_1"
        
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True # for data_dir.exists()
        mock_path_instance.__truediv__.return_value.exists.return_value = True # for parquet_path.exists()

        mock_read_parquet.side_effect = Exception("Parquet read error")
        mock_request.id = "test_task_id"
        
        with pytest.raises(Exception, match="Parquet read error"):
            validate_collected_data_task.run(mock_data_path)
        
        mock_data_tasks_logger.error.assert_called_once_with("Data validation failed: Parquet read error")

# --- Tests for check_data_freshness_task ---
def test_check_data_freshness_task_no_data_dir(mock_pipeline_dependencies, mock_data_tasks_logger):
    (_, _, _, _, _, _, _, _) = mock_pipeline_dependencies
    
    with patch('pathlib.Path') as MockPath, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        mock_request.id = "test_task_id"
        
        result = check_data_freshness_task.run()
        
        assert result["status"] == "no_data"
        assert result["needs_refresh"] is True
        mock_data_tasks_logger.info.assert_called_once_with("Checking data freshness...")
        mock_data_tasks_logger.error.assert_not_called()

def test_check_data_freshness_task_no_runs(mock_pipeline_dependencies, mock_data_tasks_logger):
    (_, _, _, _, _, _, _, _) = mock_pipeline_dependencies
    
    with patch('pathlib.Path') as MockPath, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.glob.return_value = [] # No runs found
        mock_request.id = "test_task_id"
        
        result = check_data_freshness_task.run()        
        assert result["status"] == "no_data"
        assert result["needs_refresh"] is True
        mock_data_tasks_logger.info.assert_any_call("Checking data freshness...")
        mock_data_tasks_logger.error.assert_not_called()

def test_check_data_freshness_task_fresh_data(mock_pipeline_dependencies, mock_data_tasks_logger):
    (_, _, _, _, _, _, mock_getmtime, MockDatetime) = mock_pipeline_dependencies
    
    with patch('pathlib.Path') as MockPath, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.glob.return_value = [MagicMock(name="pipeline_latest")]
        
        # Mock datetime.now() to be slightly after mtime
        latest_mtime = (datetime.now() - timedelta(hours=1)).timestamp()
        mock_getmtime.return_value = latest_mtime
        mock_request.id = "test_task_id"
        
        result = check_data_freshness_task.run()        
        assert result["status"] == "success"
        assert result["needs_refresh"] is False
        assert result["age_hours"] < 24
        mock_data_tasks_logger.info.assert_any_call("Checking data freshness...")
        mock_data_tasks_logger.error.assert_not_called()

def test_check_data_freshness_task_stale_data(mock_pipeline_dependencies, mock_data_tasks_logger):
    (_, _, _, _, _, _, mock_getmtime, MockDatetime) = mock_pipeline_dependencies
    
    with patch('pathlib.Path') as MockPath, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.glob.return_value = [MagicMock(name="pipeline_old")]
        
        # Mock datetime.now() to be significantly after mtime
        old_mtime = (datetime.now() - timedelta(days=2)).timestamp()
        mock_getmtime.return_value = old_mtime
        mock_request.id = "test_task_id"
        
        result = check_data_freshness_task.run()        
        assert result["status"] == "success"
        assert result["needs_refresh"] is True
        assert result["age_hours"] > 24
        mock_data_tasks_logger.info.assert_any_call("Checking data freshness...")
        mock_data_tasks_logger.error.assert_not_called()

def test_check_data_freshness_task_exception(mock_pipeline_dependencies, mock_data_tasks_logger):
    (_, _, _, _, _, _, _, _) = mock_pipeline_dependencies
    
    with patch('pathlib.Path') as MockPath, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        MockPath.side_effect = Exception("Path error")
        mock_request.id = "test_task_id"
        
        result = check_data_freshness_task.run()        
        assert result["status"] == "error"
        assert result["needs_refresh"] is True
        mock_data_tasks_logger.error.assert_called_once_with("Freshness check failed: Path error")

# --- Tests for scheduled_data_collection ---
def test_scheduled_data_collection_needs_refresh(mock_data_tasks_logger):
    with patch('src.tasks.data_tasks.check_data_freshness_task') as mock_freshness_task, \
         patch('src.tasks.data_tasks.collect_options_data_task') as mock_collect_task, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        
        mock_freshness_task.apply.return_value.get.return_value = {"needs_refresh": True, "message": "Data is old"}
        mock_collect_task.apply_async.return_value.id = "collection_task_123"
        mock_request.id = "test_task_id"
        
        result = scheduled_data_collection.run()
        
        mock_freshness_task.apply.assert_called_once()
        mock_collect_task.apply_async.assert_called_once()
        mock_data_tasks_logger.info.assert_any_call("Running scheduled data collection check...")
        mock_data_tasks_logger.info.assert_any_call("Data needs refresh, starting collection...")
        mock_data_tasks_logger.error.assert_not_called()
        
        assert result["status"] == "collection_started"
        assert result["collection_task_id"] == "collection_task_123"

def test_scheduled_data_collection_skips_refresh(mock_data_tasks_logger):
    with patch('src.tasks.data_tasks.check_data_freshness_task') as mock_freshness_task, \
         patch('src.tasks.data_tasks.collect_options_data_task') as mock_collect_task, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        
        mock_freshness_task.apply.return_value.get.return_value = {"needs_refresh": False, "age_hours": 1.5}
        mock_request.id = "test_task_id"
        
        result = scheduled_data_collection.run()
        
        mock_freshness_task.apply.assert_called_once()
        mock_collect_task.apply_async.assert_not_called()
        mock_data_tasks_logger.info.assert_any_call("Running scheduled data collection check...")
        mock_data_tasks_logger.info.assert_any_call("Data is fresh, skipping collection")
        mock_data_tasks_logger.error.assert_not_called()
        
        assert result["status"] == "skipped"
        assert "Data is 1.5 hours old" in result["reason"]

# --- Tests for run_full_data_pipeline_task ---
def test_run_full_data_pipeline_task_success_no_train(mock_data_tasks_logger):
    with patch('src.tasks.data_tasks.collect_options_data_task') as mock_collect_task, \
         patch('src.tasks.data_tasks.validate_collected_data_task') as mock_validate_task, \
         patch('src.tasks.ml_tasks.train_model_task') as mock_train_task, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        
        mock_collect_task.apply.return_value.get.return_value = {
            "status": "success", "output_path": "/mock/data/pipeline_1"
        }
        mock_validate_task.apply.return_value.get.return_value = {
            "validation": {"passed": True}, "output_path": "/mock/data/pipeline_1"
        }
        mock_request.id = "test_task_id"
        
        result = run_full_data_pipeline_task.run(train_after_collection=False)
        
        mock_collect_task.apply.assert_called_once()
        mock_validate_task.apply.assert_called_once()
        mock_train_task.apply_async.assert_not_called()
        mock_data_tasks_logger.info.assert_any_call("Starting full data pipeline...")
        assert result["status"] == "success"
        assert result["training_task_id"] is None

def test_run_full_data_pipeline_task_success_with_train(mock_data_tasks_logger):
    with patch('src.tasks.data_tasks.collect_options_data_task') as mock_collect_task, \
         patch('src.tasks.data_tasks.validate_collected_data_task') as mock_validate_task, \
         patch('src.tasks.ml_tasks.train_model_task') as mock_train_task, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        
        mock_collect_task.apply.return_value.get.return_value = {
            "status": "success", "output_path": "/mock/data/pipeline_1"
        }
        mock_validate_task.apply.return_value.get.return_value = {
            "validation": {"passed": True}, "output_path": "/mock/data/pipeline_1"
        }
        mock_train_task.apply_async.return_value.id = "train_task_456"
        mock_request.id = "test_task_id"
        
        result = run_full_data_pipeline_task.run(train_after_collection=True)
        
        mock_collect_task.apply.assert_called_once()
        mock_validate_task.apply.assert_called_once()
        mock_train_task.apply_async.assert_called_once()
        mock_data_tasks_logger.info.assert_any_call("Starting full data pipeline...")
        mock_data_tasks_logger.info.assert_any_call("Training triggered: train_task_456")
        assert result["status"] == "success"
        assert result["training_task_id"] == "train_task_456"

def test_run_full_data_pipeline_task_collection_failure(mock_data_tasks_logger):
    with patch('src.tasks.data_tasks.collect_options_data_task') as mock_collect_task, \
         patch('src.tasks.data_tasks.validate_collected_data_task') as mock_validate_task, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        
        mock_collect_task.apply.return_value.get.return_value = {
            "status": "failed", "error": "Collection error"
        }
        mock_request.id = "test_task_id"
        mock_request.retries = 0 # Set retries to 0 for the mock
        
        with pytest.raises(Exception, match=r"Collection failed: {'status': 'failed', 'error': 'Collection error'}"):
            run_full_data_pipeline_task.run()
        
        mock_collect_task.apply.assert_called_once()
        mock_validate_task.apply.assert_not_called()
        mock_data_tasks_logger.error.assert_called_once_with(r"Full pipeline failed: Collection failed: {'status': 'failed', 'error': 'Collection error'}")

def test_run_full_data_pipeline_task_validation_failure(mock_data_tasks_logger):
    with patch('src.tasks.data_tasks.collect_options_data_task') as mock_collect_task, \
         patch('src.tasks.data_tasks.validate_collected_data_task') as mock_validate_task, \
         patch('src.tasks.ml_tasks.train_model_task') as mock_train_task, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        
        mock_collect_task.apply.return_value.get.return_value = {
            "status": "success", "output_path": "/mock/data/pipeline_1"
        }
        mock_validate_task.apply.return_value.get.return_value = {
            "validation": {"passed": False}, "output_path": "/mock/data/pipeline_1"
        }
        mock_request.id = "test_task_id"
        mock_request.retries = 0 # Set retries to 0 for the mock
        
        result = run_full_data_pipeline_task.run(train_after_collection=True)
        
        mock_collect_task.apply.assert_called_once()
        mock_validate_task.apply.assert_called_once()
        mock_train_task.apply_async.assert_not_called() # Should not train if validation fails
        mock_data_tasks_logger.warning.assert_called_once_with("Data validation failed, proceeding with caution")
        assert result["status"] == "success"
        assert result["training_task_id"] is None

def test_run_full_data_pipeline_task_exception(mock_data_tasks_logger):
    with patch('src.tasks.data_tasks.collect_options_data_task') as mock_collect_task, \
         patch('src.tasks.data_tasks.celery_app.Task.request') as mock_request:
        mock_collect_task.apply.side_effect = Exception("Unexpected error")
        mock_request.id = "test_task_id"
        mock_request.retries = 0 # Set retries to 0 for the mock
        
        with pytest.raises(Exception, match="Unexpected error"):
            run_full_data_pipeline_task.run()
        
        mock_data_tasks_logger.error.assert_called_once_with("Full pipeline failed: Unexpected error")