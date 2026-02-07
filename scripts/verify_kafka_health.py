import sys

import requests
from confluent_kafka.admin import AdminClient


def check_kafka_brokers(bootstrap_servers):
    print(f"Checking Kafka brokers at {bootstrap_servers}...")
    try:
        admin_client = AdminClient({'bootstrap.servers': bootstrap_servers, 'socket.timeout.ms': 5000})
        metadata = admin_client.list_topics(timeout=5)
        brokers = metadata.brokers
        print(f"Successfully connected to Kafka. Found {len(brokers)} brokers.")
        for broker_id, broker in brokers.items():
            print(f" - Broker {broker_id}: {broker.host}:{broker.port}")
        return True
    except Exception as e:
        print(f"Error connecting to Kafka brokers: {e}")
        return False

def check_schema_registry(url):
    print(f"Checking Schema Registry at {url}...")
    try:
        response = requests.get(f"{url}/subjects", timeout=5)
        if response.status_code == 200:
            print("Successfully connected to Schema Registry.")
            return True
        else:
            print(f"Schema Registry returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to Schema Registry: {e}")
        return False

def check_ksqldb(url):
    print(f"Checking ksqlDB at {url}...")
    try:
        # ksqlDB /healthcheck endpoint
        response = requests.get(f"{url}/healthcheck", timeout=5)
        if response.status_code == 200:
            print("Successfully connected to ksqlDB.")
            return True
        else:
            # Try /info if /healthcheck is not available in this version
            response = requests.get(f"{url}/info", timeout=5)
            if response.status_code == 200:
                print("Successfully connected to ksqlDB (via /info).")
                return True
            print(f"ksqlDB returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to ksqlDB: {e}")
        return False

if __name__ == "__main__":
    # Note: These hostnames match the service names in docker-compose.yml
    # When running from the host, you might need to use 'localhost' if ports are mapped.
    # For this verification, we assume the user might run it from within a container or has port mapping.
    
    kafka_bootstrap = "localhost:9092" # Assuming kafka-1 is mapped to 9092 on host
    schema_registry_url = "http://localhost:8081"
    ksqldb_url = "http://localhost:8088"
    
    success = True
    if not check_kafka_brokers(kafka_bootstrap):
        success = False
    
    if not check_schema_registry(schema_registry_url):
        # Fallback to internal name if running in docker network context (though unlikely for this script)
        pass
        
    if not check_ksqldb(ksqldb_url):
        success = False
        
    if success:
        print("\nAll Kafka infrastructure components are HEALTHY!")
        sys.exit(0)
    else:
        print("\nSome Kafka infrastructure components are UNHEALTHY or UNREACHABLE.")
        sys.exit(1)
