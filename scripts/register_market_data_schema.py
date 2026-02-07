import argparse
import os

from confluent_kafka.schema_registry import Schema, SchemaRegistryClient


def register_schema(schema_path, subject_name, schema_registry_url):
    """
    Registers an Avro schema with the Confluent Schema Registry.
    """
    client = SchemaRegistryClient({"url": schema_registry_url})

    with open(schema_path) as f:
        schema_str = f.read()

    schema = Schema(schema_str, "AVRO")
    schema_id = client.register_schema(subject_name, schema)
    print(f"Schema registered with ID: {schema_id} for subject: {subject_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register Avro schema with Confluent Schema Registry."
    )
    parser.add_argument(
        "--schema-path", required=True, help="Path to the Avro schema file (.avsc)"
    )
    parser.add_argument(
        "--subject-name",
        required=True,
        help="Subject name for the schema in Schema Registry (e.g., market-data-value)",
    )
    parser.add_argument(
        "--schema-registry-url",
        default=os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081"),
        help="URL of the Schema Registry (default: http://localhost:8081)",
    )

    args = parser.parse_args()

    try:
        register_schema(args.schema_path, args.subject_name, args.schema_registry_url)
    except Exception as e:
        print(f"Error registering schema: {e}")
        exit(1)
