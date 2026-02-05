import json
import os
import sys

import requests


def register_schema(url, subject, schema_path):
    print(f"Registering schema for subject '{subject}' at {url}...")
    
    if not os.path.exists(schema_path):
        print(f"Error: Schema file not found at {schema_path}")
        return False
        
    with open(schema_path) as f:
        schema_json = json.load(f)
        
    payload = {
        "schema": json.dumps(schema_json)
    }
    
    try:
        response = requests.post(
            f"{url}/subjects/{subject}/versions",
            json=payload,
            headers={"Content-Type": "application/vnd.schemaregistry.v1+json"}
        )
        
        if response.status_code == 200:
            schema_id = response.json()["id"]
            print(f"Successfully registered schema. Schema ID: {schema_id}")
            return True
        else:
            print(f"Failed to register schema. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error connecting to Schema Registry: {e}")
        return False

if __name__ == "__main__":
    schema_registry_url = os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")
    subject_name = "market-data-value"
    avro_schema_path = "src/shared/schemas/market_data.avsc"
    
    if register_schema(schema_registry_url, subject_name, avro_schema_path):
        sys.exit(0)
    else:
        sys.exit(1)
