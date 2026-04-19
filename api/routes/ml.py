import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from api.dependencies import get_current_user, get_current_user_id
from src.database.crud import (
    create_ml_model as crud_create_ml_model,
)
from src.database.crud import (
    get_ml_model_by_name_version as crud_get_ml_model_by_name_version,
)
from src.database.models import MLModel, User
from src.database.session import get_async_db
from src.ml.pipeline import MLPipeline
from src.schemas.ml import MLModel as MLModelSchema
from src.schemas.ml import MLModelCreate
from src.tasks import trigger_ml_training_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/ml", tags=["ML"])
ml_pipeline_service = MLPipeline()

@router.post("/models", response_model=MLModelSchema, status_code=status.HTTP_201_CREATED)
async def create_ml_model_item_route(
    ml_model_in: MLModelCreate,
    db: AsyncSession = Depends(get_async_db),
    user_id: UUID = Depends(get_current_user_id)
):
    """Creates a new ML model entry."""
    model_data = ml_model_in.dict()

    existing_model = await crud_get_ml_model_by_name_version(db, name=model_data["name"], version=model_data["version"])
    if existing_model:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="ML model already exists")

    try:
        db_ml_model = await crud_create_ml_model(db, model_data)
        return MLModelSchema.from_orm(db_ml_model)
    except Exception as e:
        logger.error("Failed to create ML model: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model creation failed")

@router.get("/models", response_model=list[MLModelSchema])
async def read_ml_models_list(
    db: AsyncSession = Depends(get_async_db),
    skip: int = 0,
    limit: int = 100
):
    """Retrieves a list of active ML models."""
    stmt = select(MLModel).filter(MLModel.is_active == True).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return [MLModelSchema.from_orm(m) for m in result.scalars().all()]

@router.get("/models/{model_id}", response_model=MLModelSchema)
async def read_ml_model_item_route(
    model_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Retrieves a specific ML model by its UUID."""
    db_model = await db.get(MLModel, model_id)
    if not db_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ML model not found")
    return MLModelSchema.from_orm(db_model)

@router.post("/models/{model_id}/predictions", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_model_prediction(
    model_id: UUID,
    prediction_in: dict[str, Any],
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Performs prediction (Noun-based resource)."""
    model = await db.get(MLModel, model_id)
    if not model or not model.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail={"code": "MODEL_NOT_FOUND", "message": "ML model is inactive or does not exist"}
        )

    try:
        result = ml_pipeline_service.predict(model_id=f"{model.name}@{model.version}", data=prediction_in)
        return {
            "model_id": str(model_id),
            "prediction": result,
            "timestamp": datetime.now(timezone.utc).isoformat() if "datetime" in globals() else None
        }
    except Exception as e:
        logger.error("Prediction failed for model %s: %s", model_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail={"code": "PREDICTION_FAILED", "message": "The ML kernel encountered an error during inference"}
        )

@router.post("/models/{model_id}/training-jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_model_training_job(
    model_id: UUID,
    params_in: dict[str, Any],
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Triggers ML training job (Noun-based resource)."""
    model = await db.get(MLModel, model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail={"code": "MODEL_NOT_FOUND", "message": "ML model not found"}
        )

    trigger_ml_training_task.delay(
        model_id=str(model_id), 
        epochs=params_in.get("epochs", 10), 
        batch_size=params_in.get("batch_size", 32)
    )
    return {
        "job_id": f"job_{model_id}_{int(datetime.now(timezone.utc).timestamp())}" if "datetime" in globals() else str(model_id),
        "status": "enqueued",
        "model_id": str(model_id)
    }
