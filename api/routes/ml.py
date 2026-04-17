from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session # Needed for ORM interactions in crud/models if not fully async
from sqlalchemy.sql import select
from typing import List, Dict, Any

from src.database.session import get_async_db
from src.database.crud import (
    create_ml_model as crud_create_ml_model,
    get_ml_model_by_name_version as crud_get_ml_model_by_name_version,
    # Add other CRUD functions as needed, e.g., update_ml_model, delete_ml_model, list_ml_models
)
from src.database.models import MLModel, User
from src.schemas.ml import MLModelCreate, MLModelUpdate, MLModel as MLModelSchema # Import Pydantic schemas
from src.shared.protos import auth_pb2 # Import proto types
from src.shared.protos import auth_pb2_grpc # Import gRPC stubs

# --- Logging and Configuration ---
import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ML"])

# --- Authentication Dependency ---
async def get_current_user( # Placeholder: Real implementation from api.index.py
    request: Request, db: AsyncSession = Depends(get_async_db), auth_client: auth_pb2_grpc.AuthServiceStub = Depends(get_auth_client) 
) -> User:
    from src.database.crud import get_user_by_id
    test_user_id = "test-integration-user" 
    db_user = await get_user_by_id(db, user_id=test_user_id)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return db_user

async def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
    return current_user.id

# --- ML Model Routes ---

@router.post("/models", response_model=MLModelSchema, status_code=status.HTTP_201_CREATED) # Use schema for response
async def create_ml_model_item_route(
    ml_model_in: MLModelCreate, # Use Pydantic schema for request body
    db: AsyncSession = Depends(get_async_db),
    user_id: str = Depends(get_current_user_id) # Use authenticated user ID
):
    """Creates a new ML model entry."""
    
    model_data = ml_model_in.dict()
    model_data["user_id"] = user_id # Associate model with user if needed, though not in DB model yet

    try:
        existing_model = await crud_get_ml_model_by_name_version(db, name=model_data["name"], version=model_data["version"])
        if existing_model:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="ML model with this name and version already exists")

        db_ml_model = await crud_create_ml_model(db, model_data)
        return MLModelSchema.from_orm(db_ml_model) # Return as Pydantic schema
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to create ML model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create ML model")

@router.get("/models", response_model=List[MLModelSchema]) # Use schema for response model
async def read_ml_models_list(
    db: AsyncSession = Depends(get_async_db),
    skip: int = 0,
    limit: int = 100
):
    """Retrieves a list of active ML models."""
    stmt = select(MLModel).filter(MLModel.is_active == True).offset(skip).limit(limit)
    result = await db.execute(stmt)
    db_models = result.scalars().all()

    return [MLModelSchema.from_orm(m) for m in db_models]

@router.get("/models/{model_id}", response_model=MLModelSchema) # Use schema for response model
async def read_ml_model_item_route(
    model_id: str,
    db: AsyncSession = Depends(get_async_db)
):
    """Retrieves a specific ML model by its ID."""
    stmt = select(MLModel).filter(MLModel.id == model_id)
    result = await db.execute(stmt)
    db_model = result.scalar_one_or_none()

    if db_model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ML model not found")
    
    return MLModelSchema.from_orm(db_model)

# --- Prediction Endpoint ---
@router.post("/predict/{model_id}", response_model=Dict[str, Any])
async def predict_with_model(
    model_id: str,
    prediction_data: Dict[str, Any], # Input data for prediction
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Ensure user is authenticated
):
    """
    Performs prediction using a specified ML model.
    """
    stmt = select(MLModel).filter(MLModel.id == model_id, MLModel.is_active == True)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()

    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ML model not found or is inactive")
    
    try:
        prediction = ml_pipeline_service.predict(
            model_id=f"{model.name}@{model.version}", 
            data=prediction_data
        )
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction for model {model_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction service failed")

# --- Training Trigger Endpoint ---
@router.post("/train/{model_id}", status_code=status.HTTP_202_ACCEPTED)
async def trigger_training(
    model_id: str,
    training_params: Dict[str, Any], # e.g., {"epochs": 50, "batch_size": 128}
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Ensure user is authenticated
):
    """
    Triggers model training asynchronously using Celery.
    """
    stmt = select(MLModel).filter(MLModel.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ML model not found")

    epochs = training_params.get("epochs", 10)
    batch_size = training_params.get("batch_size", 32)

    try:
        # Import Celery task here to avoid potential circular dependencies or import issues
        from src.tasks import trigger_ml_training_task 
        trigger_ml_training_task.delay(model_id=model_id, epochs=epochs, batch_size=batch_size)
        return {"message": "ML training task enqueued successfully", "model_id": model_id, "status": "accepted"}
    except Exception as e:
        logger.error(f"Failed to enqueue ML training task for model {model_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to enqueue ML training task")

