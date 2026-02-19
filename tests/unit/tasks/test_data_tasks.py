from unittest.mock import patch

import pytest

from src.tasks.data_tasks import collect_options_data_task


@patch("src.data.pipeline.DataPipeline.run")
def test_collect_options_data_task_success(mock_run):
    # Mock pipeline report
    mock_run.return_value = {
        "samples_collected": 1000,
        "samples_valid": 950,
        "output_path": "data/test",
        "duration_seconds": 10.5,
        "validation_rate": 0.95,
    }

    result = collect_options_data_task.__wrapped__(["AAPL"])

    assert result["status"] == "success"
    assert result["samples_valid"] == 950


@patch("src.data.pipeline.DataPipeline.run")
def test_collect_options_data_task_failure(mock_run):
    mock_run.side_effect = Exception("Pipeline crashed")

    with pytest.raises(Exception, match="Pipeline crashed"):
        collect_options_data_task.__wrapped__(["AAPL"])
