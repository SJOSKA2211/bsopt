from unittest.mock import MagicMock, patch

import pytest

from src.ml.celery_app import run_pipeline_task


@patch("src.ml.celery_app.AutonomousMLPipeline")
def test_run_pipeline_task_success(mock_pipeline_class):
    """Verify that the Celery task runs the pipeline successfully."""
    # Mock pipeline instance and its run method
    mock_pipeline = mock_pipeline_class.return_value
    mock_study = MagicMock()
    mock_study.best_value = 0.95
    mock_study.best_params = {"n_estimators": 100}
    mock_pipeline.run.return_value = mock_study

    # Define test config
    config = {
        "ticker": "GOOGL",
        "api_key": "test_key",
        "db_url": "sqlite:///:memory:",
        "study_name": "test_celery",
        "n_trials": 1,
    }

    # Run the task directly (synchronously for testing)
    result = run_pipeline_task(config)

    # Assertions
    mock_pipeline_class.assert_called_with(config)
    mock_pipeline.run.assert_called_once()
    assert result["status"] == "success"
    assert result["best_value"] == 0.95
    assert result["best_params"] == {"n_estimators": 100}


@patch("src.ml.celery_app.AutonomousMLPipeline")
def test_run_pipeline_task_failure(mock_pipeline_class):
    """Verify that the Celery task handles failures correctly."""
    # Mock pipeline to raise exception
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.run.side_effect = Exception("Pipeline crashed")

    config = {"ticker": "FAIL"}

    with pytest.raises(Exception, match="Pipeline crashed"):
        run_pipeline_task(config)
