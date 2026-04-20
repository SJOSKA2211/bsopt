import time  # For timestamping test data
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

# Assuming api_client, db_session, test_user_token, auth_headers fixtures are available from conftest.py
# Import necessary models and schemas
from src.database.models import MLModel  # For direct DB checks

# Base URL for the API service
API_URL = "http://localhost:8000/api/v1"

pytestmark = pytest.mark.integration

# --- Helper Functions ---
async def create_test_ml_model_via_api(api_client: AsyncClient, auth_headers: dict[str, str], model_data: dict[str, Any]) -> dict[str, Any]:
    """Helper to create an ML model via the API."""
    response = await api_client.post("/api/v1/ml/models/", json=model_data, headers=auth_headers)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    return response.json()

# --- Tests ---

@pytest.mark.asyncio
async def test_create_and_get_ml_model(api_client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession):
    """Tests creating and retrieving an ML model via API."""
    timestamp_suffix = str(int(time.time()))
    model_name = f"TestModel {timestamp_suffix}"
    model_version = "1.0.0"
    model_description = "An ML model for testing purposes"

    model_create_data = {
        "name": model_name,
        "version": model_version,
        "description": model_description,
        "is_active": True,
    }

    # Test create
    response_create = await api_client.post("/api/v1/ml/models/", json=model_create_data, headers=auth_headers)
    assert response_create.status_code == 201
    created_model = response_create.json()

    assert created_model["name"] == model_name
    assert created_model["version"] == model_version
    assert created_model["description"] == model_description
    assert created_model["is_active"] is True
    assert created_model["id"] is not None

    model_id = created_model["id"]

    # Test get by ID
    response_get = await api_client.get(f"/api/v1/ml/models/{model_id}", headers=auth_headers)
    assert response_get.status_code == 200
    retrieved_model = response_get.json()

    assert retrieved_model["id"] == model_id
    assert retrieved_model["name"] == model_name
    assert retrieved_model["version"] == model_version
    assert retrieved_model["description"] == model_description
    assert retrieved_model["is_active"] is True

async def test_get_ml_model_not_found(api_client: AsyncClient, auth_headers: dict[str, str]):
    """Tests retrieving a non-existent ML model."""
    non_existent_id = "non-existent-ml-model-id"
    response = await api_client.get(f"/api/v1/ml/models/{non_existent_id}", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "ML model not found"

async def test_list_ml_models(api_client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession):
    """Tests listing active ML models."""
    # Create a couple of models to ensure listing works
    model1_name = f"ListModel1 {int(time.time())}"
    model2_name = f"ListModel2 {int(time.time())}"

    model1_data = {"name": model1_name, "version": "1.0.0", "is_active": True}
    model2_data = {"name": model2_name, "version": "1.1.0", "is_active": True}
    model3_data = {"name": "InactiveModel", "version": "1.0.0", "is_active": False} # Inactive model

    # Manually create in DB to ensure they exist and have specific states
    m1 = MLModel(**model1_data)
    m2 = MLModel(**model2_data)
    m3 = MLModel(**model3_data)
    db_session.add_all([m1, m2, m3])
    await db_session.commit()
    await db_session.refresh(m1)
    await db_session.refresh(m2)
    await db_session.refresh(m3)

    response = await api_client.get("/api/v1/ml/models", headers=auth_headers)

    assert response.status_code == 200
    models = response.json()

    # Check if the created ACTIVE models are in the list
    found_m1 = any(m["name"] == model1_name and m["version"] == "1.0.0" and m["is_active"] is True for m in models)
    found_m2 = any(m["name"] == model2_name and m["version"] == "1.1.0" and m["is_active"] is True for m in models)
    found_m3 = any(m["name"] == "InactiveModel" for m in models) # Should not be found if filtering works

    assert found_m1
    assert found_m2
    assert not found_m3 # Ensure inactive model is not listed

@pytest.mark.asyncio
async def test_predict_with_model(api_client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession):
    """Tests the ML prediction endpoint."""
    # Create a model first
    model_name = f"PredictModel {int(time.time())}"
    model_version = "2.0.0"
    model_data = {"name": model_name, "version": model_version, "is_active": True}

    created_model = await create_test_ml_model_via_api(api_client, auth_headers, model_data)
    model_id = created_model["id"]

    prediction_input = {"input_value": 42.5, "config": {"param": "value"}}

    response = await api_client.post(f"/api/v1/ml/predict/{model_id}", json=prediction_input, headers=auth_headers)

    assert response.status_code == 200
    prediction_output = response.json()

    assert "prediction" in prediction_output
    assert "confidence" in prediction_output
    assert prediction_output["model_used"] == f"{model_name}@{model_version}"
    assert "timestamp" in prediction_output

async def test_predict_model_not_found(api_client: AsyncClient, auth_headers: dict[str, str]):
    """Tests prediction with a non-existent model ID."""
    non_existent_model_id = "non-existent-model-123"
    prediction_input = {"input_value": 10.0}

    response = await api_client.post(f"/api/v1/ml/predict/{non_existent_model_id}", json=prediction_input, headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "ML model not found or is inactive"

@pytest.mark.asyncio
async def test_trigger_ml_training(api_client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession):
    """Tests triggering ML model training via API."""
    # Create a model first to have a valid model_id
    model_name = f"TrainModel {int(time.time())}"
    model_version = "3.0.0"
    model_data = {"name": model_name, "version": model_version, "is_active": True}

    created_model = await create_test_ml_model_via_api(api_client, auth_headers, model_data)
    model_id = created_model["id"]

    training_params = {"epochs": 50, "batch_size": 128}

    response = await api_client.post(f"/api/v1/ml/train/{model_id}", json=training_params, headers=auth_headers)

    assert response.status_code == 202 # Accepted
    task_info = response.json()

    assert task_info["message"] == "ML training task enqueued successfully"
    assert task_info["model_id"] == model_id
    assert task_info["training_parameters"]["epochs"] == 50
    assert task_info["training_parameters"]["batch_size"] == 128
    assert task_info["status"] == "queued"
    assert "timestamp" in task_info

async def test_trigger_training_model_not_found(api_client: AsyncClient, auth_headers: dict[str, str]):
    """Tests triggering training for a non-existent ML model."""
    non_existent_model_id = "non-existent-model-id-for-training"
    training_params = {"epochs": 10}

    response = await api_client.post(f"/api/v1/ml/train/{non_existent_model_id}", json=training_params, headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "ML model not found"

@pytest.mark.asyncio
async def test_deploy_ml_model(api_client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession):
    """Tests triggering ML model deployment via API."""
    # Create a model first to have a valid model_id
    model_name = f"DeployModel {int(time.time())}"
    model_version = "1.0.0"
    model_data = {"name": model_name, "version": model_version, "is_active": True}

    created_model = await create_test_ml_model_via_api(api_client, auth_headers, model_data)
    model_id = created_model["id"]

    deployment_params = {"version": model_version, "target_environment": "staging"}

    response = await api_client.post(f"/api/v1/ml/deploy/{model_id}", json=deployment_params, headers=auth_headers)

    assert response.status_code == 202 # Accepted
    deployment_info = response.json()

    assert deployment_info["message"] == "ML model deployment task enqueued successfully"
    assert deployment_info["model_id"] == model_id
    assert deployment_info["version"] == model_version
    assert deployment_info["target_environment"] == "staging"
    assert deployment_info["status"] == "queued"
    assert "timestamp" in deployment_info

async def test_deploy_model_not_found(api_client: AsyncClient, auth_headers: dict[str, str]):
    """Tests triggering deployment for a non-existent ML model."""
    non_existent_model_id = "non-existent-model-id-for-deploy"
    deployment_params = {"version": "1.0.0", "target_environment": "staging"}

    response = await api_client.post(f"/api/v1/ml/deploy/{non_existent_model_id}", json=deployment_params, headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "ML model not found"

# Note: Testing Celery task execution itself (e.g., checking result backend) is more complex
# and might require specific setup or mocking of Celery workers for unit/integration tests.
# These tests focus on the API layer correctly enqueueing the tasks.
