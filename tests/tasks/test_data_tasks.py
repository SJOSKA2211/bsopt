import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.tasks.data_tasks import (
    collect_options_data_task,
    validate_collected_data_task,
    check_data_freshness_task,
    scheduled_data_collection,
    run_full_data_pipeline_task
)

@pytest.fixture
def mock_celery_eager():
    from src.tasks.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    return celery_app

@pytest.fixture
def mock_data_pipeline():
    with patch("src.data.pipeline.DataPipeline") as MockPipeline:
        pipeline = MagicMock()
        MockPipeline.return_value = pipeline
        
        import asyncio
        async def async_run():
            return {
                "samples_collected": 100,
                "samples_valid": 90,
                "output_path": "/tmp/data",
                "duration_seconds": 10,
                "validation_rate": 0.9
            }
        
        pipeline.run.side_effect = async_run
        yield MockPipeline

def test_collect_options_data_task(mock_celery_eager, mock_data_pipeline):
    res = collect_options_data_task.apply(kwargs={"symbols": ["AAPL"]})
    result = res.get()
    
    assert result["status"] == "success"
    assert result["samples_collected"] == 100
    mock_data_pipeline.assert_called()

def test_collect_options_data_error(mock_celery_eager):
    with patch("src.data.pipeline.DataPipeline", side_effect=Exception("Pipeline Error")):
        with pytest.raises(Exception):
            collect_options_data_task.apply().get()

def test_validate_collected_data_task(mock_celery_eager, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    parquet_path = data_dir / "training_data.parquet"
    
    df = pd.DataFrame({
        "feature1": np.random.rand(100),
        "target": np.random.rand(100)
    })
    df.to_parquet(parquet_path)
    
    res = validate_collected_data_task.apply(args=[str(data_dir)])
    result = res.get()
    
    assert result["status"] == "success"
    assert result["validation"]["passed"] is True

def test_validate_collected_data_missing_file(mock_celery_eager, tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_collected_data_task.apply(args=[str(tmp_path)]).get()

def test_validate_collected_data_poor_quality(mock_celery_eager, tmp_path):
    data_dir = tmp_path / "data_poor"
    data_dir.mkdir()
    parquet_path = data_dir / "training_data.parquet"
    
    df = pd.DataFrame({
        "feature1": [np.nan] * 50 + [1.0] * 50,
        "target": [1.0] * 100
    })
    df = pd.concat([df, df]) 
    df.to_parquet(parquet_path)
    
    res = validate_collected_data_task.apply(args=[str(data_dir)])
    result = res.get()
    
    assert result["status"] == "success"
    assert result["validation"]["quality_score"] < 1.0

def test_check_data_freshness_no_dir(mock_celery_eager):
    with patch("pathlib.Path") as MockPath:
        MockPath.return_value.exists.return_value = False
        
        res = check_data_freshness_task.apply()
        result = res.get()
        assert result["status"] == "no_data"
        assert result["needs_refresh"] is True

def test_check_data_freshness_fresh(mock_celery_eager):
    with patch("pathlib.Path") as MockPath, \
         patch("os.path.getmtime") as mock_mtime:
        
        mock_dir = MagicMock()
        MockPath.return_value = mock_dir # "data/training"
        mock_dir.exists.return_value = True
        
        mock_run = MagicMock()
        mock_run.name = "pipeline_run_1"
        mock_dir.glob.return_value = [mock_run]
        
        from datetime import datetime
        mock_mtime.return_value = datetime.now().timestamp() - 3600 # 1 hour ago
        
        res = check_data_freshness_task.apply()
        result = res.get()
        assert result["status"] == "success"
        assert result["needs_refresh"] is False

def test_scheduled_data_collection_needs_refresh(mock_celery_eager):
    with patch("src.tasks.data_tasks.check_data_freshness_task.apply") as mock_check, \
         patch("src.tasks.data_tasks.collect_options_data_task.apply_async") as mock_collect:
        
        mock_check.return_value.get.return_value = {"needs_refresh": True, "message": "Old"}
        mock_collect.return_value.id = "task-123"
        
        res = scheduled_data_collection.apply()
        result = res.get()
        
        assert result["status"] == "collection_started"
        mock_collect.assert_called()

def test_run_full_data_pipeline_task(mock_celery_eager):
    with patch("src.tasks.data_tasks.collect_options_data_task.apply") as mock_collect, \
         patch("src.tasks.data_tasks.validate_collected_data_task.apply") as mock_validate, \
         patch("src.tasks.ml_tasks.train_model_task.apply_async") as mock_train:
        
        mock_collect.return_value.get.return_value = {
            "status": "success", 
            "output_path": "path"
        }
        
        mock_validate.return_value.get.return_value = {"validation": {"passed": True}}
        mock_train.return_value.id = "train-1"
        
        # Mocking ml_tasks module
        import sys
        mock_ml_tasks = MagicMock()
        mock_ml_tasks.train_model_task = MagicMock()
        mock_ml_tasks.train_model_task.apply_async = mock_train
        
        with patch.dict(sys.modules, {"src.tasks.ml_tasks": mock_ml_tasks}):
            res = run_full_data_pipeline_task.apply()
            result = res.get()
            
            assert result["status"] == "success"

def test_full_pipeline_fail(mock_celery_eager):
    with patch("src.tasks.data_tasks.collect_options_data_task.apply") as mock_collect:
        mock_collect.return_value.get.return_value = {"status": "failed"}
        
        with pytest.raises(Exception):
            run_full_data_pipeline_task.apply().get()