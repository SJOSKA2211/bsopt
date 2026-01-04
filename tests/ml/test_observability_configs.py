import json
import os
import yaml
import pytest

def test_prometheus_rules_valid_yaml():
    """Verify that Prometheus rules file is valid YAML."""
    path = "src/shared/observability_configs/prometheus_rules.yml"
    assert os.path.exists(path)
    with open(path, "r") as f:
        try:
            data = yaml.safe_load(f)
            assert "groups" in data
            assert len(data["groups"]) > 0
            assert data["groups"][0]["name"] == "ml_pipeline_alerts"
        except yaml.YAMLError:
            pytest.fail("Prometheus rules file is not valid YAML")

def test_grafana_dashboard_valid_json():
    """Verify that Grafana dashboard file is valid JSON."""
    path = "src/shared/observability_configs/grafana_dashboard.json"
    assert os.path.exists(path)
    with open(path, "r") as f:
        try:
            data = json.load(f)
            assert data["title"] == "BS-Opt ML Pipeline Dashboard"
            assert "panels" in data
            assert len(data["panels"]) >= 4
        except json.JSONDecodeError:
            pytest.fail("Grafana dashboard file is not valid JSON")
