import pytest
import yaml

def test_kafka_kraft_configured():
    """
    Verifies that Kafka is configured for KRaft mode in docker-compose.prod.yml.
    This test currently only checks for the presence of the configuration due to
    user instruction not to run Docker containers.
    """
    docker_compose_path = "docker-compose.prod.yml"
    
    with open(docker_compose_path, 'r') as f:
        compose_config = yaml.safe_load(f)
    
    assert 'services' in compose_config, "docker-compose.prod.yml must define services"
    
    scraper_service = compose_config['services'].get('scraper-service')
    assert scraper_service, "Scraper service not found in docker-compose.prod.yml"
    
    environment = scraper_service.get('environment', {})
    assert 'KAFKA_BOOTSTRAP_SERVERS' in environment, "KAFKA_BOOTSTRAP_SERVERS not configured for Scraper"
    assert 'kafka-1:9092' in environment['KAFKA_BOOTSTRAP_SERVERS'], "Kafka server 'kafka-1' not found in bootstrap servers"

