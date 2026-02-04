import pytest
import yaml

def test_blockchain_quantum_configured():
    """
    Verifies that Geth and Quantum Simulator services are configured in docker-compose.prod.yml.
    This test currently only checks for the presence of the configuration due to
    user instruction not to run Docker containers.
    """
    docker_compose_path = "docker-compose.prod.yml"
    
    with open(docker_compose_path, 'r') as f:
        compose_config = yaml.safe_load(f)
    
    assert 'services' in compose_config, "docker-compose.prod.yml must define services"
    
    geth_service = compose_config['services'].get('geth')
    assert geth_service, "Geth service not found in docker-compose.prod.yml"
    assert 'image' in geth_service and 'ethereum/client-go' in geth_service['image'], "Geth image not correctly specified"
    assert 'networks' in geth_service and 'blockchain-net' in geth_service['networks'], "Geth network not correctly specified"

    quantum_simulator_service = compose_config['services'].get('quantum-simulator')
    assert quantum_simulator_service, "Quantum Simulator service not found in docker-compose.prod.yml"
    assert 'build' in quantum_simulator_service, "Quantum Simulator build context not specified"
    assert 'networks' in quantum_simulator_service and 'app-net' in quantum_simulator_service['networks'], "Quantum Simulator network not correctly specified"
