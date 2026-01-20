import json
import os
import pytest
from src.utils.dashboard import generate_html_dashboard

def test_generate_html_dashboard(tmp_path):
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "dashboard.html"
    
    summary_data = {
        "timestamp": "2026-01-17 12:00:00",
        "status": "success",
        "xgboost": {"r2": 0.99, "mse": 0.01},
        "neural_network": {"accuracy": 0.95, "precision": 0.94},
        "mlflow_tracking_uri": "http://localhost:5000"
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary_data, f)
        
    generate_html_dashboard(str(summary_path), str(output_path))
    
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
        assert "ML Performance Dashboard" in content
        assert "0.9900" in content
        assert "0.9500" in content