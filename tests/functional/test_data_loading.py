"""
Data Loading Functional Tests (Principles 62, 70, 78, 86)
======================================================
"""

import json
import uuid

import pandas as pd
import pytest
import yaml


def load_csv_data():
    df = pd.read_csv("tests/test_data.csv")
    return df.to_dict(orient="records")

def load_json_data():
    with open("tests/test_data.json") as f:
        return json.load(f)

def load_yaml_data():
    with open("tests/test_data.yaml") as f:
        return yaml.safe_load(f)

@pytest.mark.asyncio
@pytest.mark.parametrize("user_data", load_csv_data())
async def test_register_from_csv(client, user_data):
    """62. Test Data: Use CSV files."""
    user_data["email"] = f"csv_{uuid.uuid4().hex[:6]}_{user_data['email']}"
    user_data["password_confirm"] = user_data["password"]
    user_data["accept_terms"] = True
    response = await client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 201

@pytest.mark.asyncio
@pytest.mark.parametrize("user_data", load_json_data())
async def test_register_from_json(client, user_data):
    """70. Test Data: Use JSON files."""
    user_data["email"] = f"json_{uuid.uuid4().hex[:6]}_{user_data['email']}"
    user_data["password_confirm"] = user_data["password"]
    user_data["accept_terms"] = True
    response = await client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 201

@pytest.mark.asyncio
@pytest.mark.parametrize("user_data", load_yaml_data())
async def test_register_from_yaml(client, user_data):
    """78. Test Data: Use YAML files."""
    user_data["email"] = f"yaml_{uuid.uuid4().hex[:6]}_{user_data['email']}"
    user_data["password_confirm"] = user_data["password"]
    user_data["accept_terms"] = True
    response = await client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 201
