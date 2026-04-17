from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any

from src.database.session import get_async_db
from src.database.crud import (
    create_ml_model as crud_create_ml_model,
    get_ml_model_by_name_version as crud_get_ml_model_by_name_version,
    # Add other CRUD functions as needed
)
from src.database.models import MLModel, User
from src.schemas.ml import MLModelCreate, MLModelUpdate, MLModel as MLModelSchema # Import Pydantic schemas
from src.ml.pipeline import MLPipeline
from src.tasks import deploy_ml_model_task # Import the new Celery task

# --- Service Instances ---
ml_pipeline_service = MLPipeline()

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

router = APIRouter(prefix="/api/v1/ml", tags=["ML"])

@router.post("/models", response_model=MLModelSchema, status_code=status.HTTP_201_CREATED)
async def create_ml_model_item_route(
    ml_model_in: MLModelCreate,
    db: AsyncSession = Depends(get_async_db),
    user_id: str = Depends(get_current_user_id) 
):
    """Creates a new ML model entry."""
    model_data = ml_model_in.dict()
    model_data["user_id"] = user_id # Associate model with user if needed

    try:
        existing_model = await crud_get_ml_model_by_name_version(db, name=model_data["name"], version=model_data["version"])
        if existing_model:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="ML model with this name and version already exists")

        db_ml_model = await crud_create_ml_model(db, model_data)
        return MLModelSchema.from_orm(db_ml_model)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to create ML model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create ML model")

@router.get("/models", response_model=List[MLModelSchema])
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

@router.get("/models/{model_id}", response_model=MLModelSchema)
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
        # Assumes src.tasks.trigger_ml_training_task is correctly defined and importable.
        # from src.tasks import trigger_ml_training_task # This import is already at the top level of the file
        trigger_ml_training_task.delay(model_id=model_id, epochs=epochs, batch_size=batch_size)
        return {"message": "ML training task enqueued successfully", "model_id": model_id, "status": "accepted"}
    except Exception as e:
        logger.error(f"Failed to enqueue ML training task for model {model_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to enqueue ML training task")

# --- New Endpoint: Deploy ML Model ---
@router.post("/deploy/{model_id}", status_code=status.HTTP_202_ACCEPTED)
async def deploy_ml_model_endpoint(
    model_id: str,
    deployment_params: Dict[str, Any], # e.g., {"version": "1.1.0", "target_environment": "production"}
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user) # Ensure user is authenticated
):
    """
    Triggers ML model deployment asynchronously using Celery.
    """
    # Validate model exists and retrieve version
    stmt = select(MLModel).filter(MLModel.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()

    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ML model not found")
    
    version_to_deploy = deployment_params.get("version", model.version) # Use provided version or current active version
    target_env = deployment_params.get("target_environment", "staging") # Default to staging

    try:
        deploy_ml_model_task.delay(model_id=model_id, version=version_to deploy, target_environment=target_env)
        return {
            "message": "ML model deployment task enqueued successfully", 
            "model_id": model_id, 
            "version": version_to_deploy,
            "target_environment": target_env,
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Failed to enqueue ML model deployment task for model {model_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to enqueue ML model deployment task")
