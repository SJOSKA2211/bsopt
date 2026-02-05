import os

import pytest
import yaml


@pytest.fixture
def prod_compose_config():
    """Load the production docker-compose file."""
    compose_path = "docker-compose.prod.yml"
    if not os.path.exists(compose_path):
        pytest.fail(f"{compose_path} not found")
    
    with open(compose_path) as f:
        return yaml.safe_load(f)

def test_neural_pricing_resource_pinning(prod_compose_config):
    """Test that neural-pricing service has CPU pinning (cpuset) configured."""
    services = prod_compose_config.get("services", {})
    assert "neural-pricing" in services, "neural-pricing service missing from prod compose"
    
    pricing_service = services["neural-pricing"]
    # Check for cpuset configuration (Kernel Bypass/Locality optimization)
    assert "cpuset" in pricing_service, "cpuset (CPU pinning) not configured for neural-pricing"
    assert pricing_service["cpuset"] == "0-1", "neural-pricing should be pinned to specific cores"

def test_neural_pricing_thread_concurrency(prod_compose_config):
    """Test that thread contention is prevented via environment variables."""
    services = prod_compose_config.get("services", {})
    pricing_service = services.get("neural-pricing", {})
    
    env = pricing_service.get("environment", {})
    if isinstance(env, list):
        # Convert list format to dict for easier testing
        env_dict = {}
        for item in env:
            if "=" in item:
                k, v = item.split("=", 1)
                env_dict[k] = v
        env = env_dict

    assert env.get("OMP_NUM_THREADS") == "1", "OMP_NUM_THREADS should be 1 to prevent contention"
    assert env.get("MKL_NUM_THREADS") == "1", "MKL_NUM_THREADS should be 1 to prevent contention"

def test_model_quantization_env(prod_compose_config):
    """Test that quantization flags are enabled for neural services."""
    services = prod_compose_config.get("services", {})
    pricing_service = services.get("neural-pricing", {})
    
    env = pricing_service.get("environment", {})
    if isinstance(env, list):
        env_dict = {}
        for item in env:
            if "=" in item:
                k, v = item.split("=", 1)
                env_dict[k] = v
        env = env_dict

    assert env.get("USE_QUANTIZED_MODELS") == "true", "Model quantization should be enabled in prod"
