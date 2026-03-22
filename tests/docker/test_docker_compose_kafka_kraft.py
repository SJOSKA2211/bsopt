import yaml


def test_kafka_kraft_configured():
    """
    Verifies that Kafka is configured for KRaft mode in docker-compose.prod.yml.
    This test currently only checks for the presence of the configuration due to
    user instruction not to run Docker containers.
    """
    docker_compose_path = "docker-compose.prod.yml"
    
    with open(docker_compose_path) as f:
        compose_config = yaml.safe_load(f)
    
    assert 'services' in compose_config, "docker-compose.prod.yml must define services"
    
    kafka_service = compose_config['services'].get('kafka-1')
    assert kafka_service, "Kafka service 'kafka-1' not found in docker-compose.prod.yml"
    
    environment = kafka_service.get('environment', {})
    assert 'KAFKA_PROCESS_ROLES' in environment, "KAFKA_PROCESS_ROLES not configured for Kafka"
    assert 'broker,controller' in environment['KAFKA_PROCESS_ROLES'], "KAFKA_PROCESS_ROLES not set to 'broker,controller' for KRaft mode"
    assert 'KAFKA_NODE_ID' in environment, "KAFKA_NODE_ID not configured for Kafka"
    assert 'KAFKA_CONTROLLER_QUORUM_VOTERS' in environment, "KAFKA_CONTROLLER_QUORUM_VOTERS not configured for Kafka"

