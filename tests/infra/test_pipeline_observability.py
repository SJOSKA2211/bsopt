import os
import yaml
import pytest

def test_github_exporter_in_docker_compose():
    """Verify that github-exporter is in the production docker-compose file."""
    with open("docker-compose.prod.yml", "r") as f:
        config = yaml.safe_load(f)
    
    assert "github-exporter" in config["services"]
    exporter = config["services"]["github-exporter"]
    assert "caarlos0/github-exporter" in exporter["image"]
    assert "9101:9101" in exporter["ports"]

def test_prometheus_scrapes_github_exporter():
    """Verify that Prometheus is configured to scrape github-exporter."""
    with open("monitoring/prometheus/prometheus.yml", "r") as f:
        config = yaml.safe_load(f)
    
    jobs = [job["job_name"] for job in config["scrape_configs"]]
    assert "github-exporter" in jobs
    
    # Check targets
    for job in config["scrape_configs"]:
        if job["job_name"] == "github-exporter":
            assert job["static_configs"][0]["targets"] == ["github-exporter:9101"]

def test_grafana_pipeline_dashboard_exists():
    """Verify that the Grafana pipeline dashboard file exists."""
    assert os.path.exists("monitoring/grafana/dashboards/pipeline-metrics.json")
    with open("monitoring/grafana/dashboards/pipeline-metrics.json", "r") as f:
        content = f.read()
    
    assert "BS-Opt Pipeline Observability" in content
    assert "Build Success Rate" in content
    assert "github_workflow_run_status" in content
