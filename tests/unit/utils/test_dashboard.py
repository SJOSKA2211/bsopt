import pytest
import json
import os
from src.utils.dashboard import generate_html_dashboard

def test_generate_html_dashboard(tmp_path):
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "dashboard.html"
    
    summary = {
        "timestamp": "2025-01-01",
        "status": "healthy",
        "xgboost": {"r2": 0.95, "mse": 0.01},
        "neural_network": {"accuracy": 0.9, "precision": 0.88},
        "mlflow_tracking_uri": "http://localhost:5000"
    }
    
    summary_path.write_text(json.dumps(summary))
    generate_html_dashboard(str(summary_path), str(output_path))
    
    assert output_path.exists()
    content = output_path.read_text()
    assert "ML Performance Dashboard" in content
    assert "0.9500" in content
