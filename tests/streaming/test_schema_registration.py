import os
import json
import pytest
from unittest.mock import MagicMock, patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))
from register_market_data_schema import register_schema
from confluent_kafka.schema_registry import Schema

SCHEMA_PATH = "src/streaming/schemas/market_data.avsc"
TEST_SUBJECT_NAME = "test-market-data-value"
TEST_SCHEMA_REGISTRY_URL = "http://mock-registry:8081"

def test_market_data_schema_exists():
    """
    Test that the market_data.avsc schema file exists.
    """
    assert os.path.exists(SCHEMA_PATH), f"Schema file not found at {SCHEMA_PATH}"

def test_market_data_schema_is_valid_json():
    """
    Test that the market_data.avsc schema file is valid JSON.
    """
    with open(SCHEMA_PATH, 'r') as f:
        schema_content = f.read()
    try:
        json.loads(schema_content)
    except json.JSONDecodeError as e:
        pytest.fail(f"Schema file {SCHEMA_PATH} is not valid JSON: {e}")

@patch('register_market_data_schema.SchemaRegistryClient')
def test_register_market_data_schema_script_interaction(mock_schema_registry_client):
    """
    Test that the register_schema function correctly interacts with a mocked
    SchemaRegistryClient to register the schema.
    """
    mock_client_instance = MagicMock()
    mock_schema_registry_client.return_value = mock_client_instance
    mock_client_instance.register_schema.return_value = 1 # Simulate a successful registration

    with open(SCHEMA_PATH, 'r') as f:
        expected_schema_str = f.read()

    register_schema(SCHEMA_PATH, TEST_SUBJECT_NAME, TEST_SCHEMA_REGISTRY_URL)

    # Assert that SchemaRegistryClient was instantiated with the correct URL
    mock_schema_registry_client.assert_called_once_with({'url': TEST_SCHEMA_REGISTRY_URL})

    # Assert that the register method was called on the client instance
    args, kwargs = mock_client_instance.register.call_args
    registered_subject_name = args[0]
    registered_schema_obj = args[1]

    assert registered_subject_name == TEST_SUBJECT_NAME
    assert isinstance(registered_schema_obj, Schema)
    assert registered_schema_obj.schema_str == expected_schema_str