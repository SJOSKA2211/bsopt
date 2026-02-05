import json
from unittest.mock import mock_open, patch

from src.utils.dashboard import generate_html_dashboard


def test_generate_html_dashboard():
    summary_data = {
        "timestamp": "2023-01-01 12:00:00",
        "status": "success",
        "xgboost": {"r2": 0.95, "mse": 0.01},
        "neural_network": {"accuracy": 0.98, "precision": 0.97},
        "mlflow_tracking_uri": "http://mlflow"
    }
    
    mock_file = mock_open(read_data=json.dumps(summary_data))
    
    with patch("builtins.open", mock_file):
        generate_html_dashboard("summary.json", "output.html")
    
    # Check if files were opened correctly
    assert mock_file.call_count == 2
    mock_file.assert_any_call("summary.json", "r")
    mock_file.assert_any_call("output.html", "w")
    
    # Check if write was called with HTML content
    handle = mock_file()
    handle.write.assert_called()
    args = handle.write.call_args[0][0]
    assert "<!DOCTYPE html>" in args
    assert "0.9500" in args