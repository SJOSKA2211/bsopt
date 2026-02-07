import os

import pytest
import yaml


def test_docker_neural_pricing_optimization():
    compose_path = "docker-compose.prod.yml"
    if not os.path.exists(compose_path):
        pytest.skip("docker-compose.prod.yml not found")
        
    with open(compose_path) as f:
        config = yaml.safe_load(f)
        
    services = config.get("services", {})
    assert "neural-pricing" in services, "neural-pricing service should be defined"
    
    np_service = services["neural-pricing"]
    
    # Check for core pinning
    # Note: cpuset is usually in deploy.resources.reservations or similar in v3, 
    # but in compose it can be top level
    assert "cpuset" in np_service or "cpuset" in np_service.get("deploy", {}).get("resources", {}).get("reservations", {}), \
        "CPU core pinning (cpuset) should be configured"
        
    # Check for environment variables
    env = np_service.get("environment", [])
    if isinstance(env, list):
        env_dict = {item.split('=')[0]: item.split('=')[1] for item in env}
    else:
        env_dict = env
        
    assert env_dict.get("OMP_NUM_THREADS") == "1", "OMP_NUM_THREADS should be 1"
    assert env_dict.get("MKL_NUM_THREADS") == "1", "MKL_NUM_THREADS should be 1"
